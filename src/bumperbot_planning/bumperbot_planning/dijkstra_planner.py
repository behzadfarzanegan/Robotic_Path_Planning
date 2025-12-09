#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from rclpy.time import Time
from nav_msgs.msg import OccupancyGrid, Path
from geometry_msgs.msg import PoseStamped, Pose
from rclpy.qos import QoSProfile, DurabilityPolicy
from tf2_ros import Buffer, TransformListener
from queue import PriorityQueue
import time

class GraphNode:
    def __init__(self, x, y, cost=0, prev=None):
        self.x = int(x)
        self.y = int(y)
        self.cost = cost
        self.prev = prev

    def __lt__(self, other):
        return self.cost < other.cost

    def __eq__(self, other):
        return isinstance(other, GraphNode) and self.x == other.x and self.y == other.y

    def __hash__(self):
        return hash((self.x, self.y))

    def __add__(self, other):
        return GraphNode(self.x + other[0], self.y + other[1])

class DijkstraPlanner(Node):
    def __init__(self):
        super().__init__("dijkstra_node")
        self.tf_buffer = Buffer()
        self.tf_listener = TransformListener(self.tf_buffer, self)

        # map subscription: transient local so we get latched map
        map_qos = QoSProfile(depth=10)
        map_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_sub = self.create_subscription(
            OccupancyGrid, "/map", self.map_callback, map_qos
        )

        self.pose_sub = self.create_subscription(
            PoseStamped, "/goal_pose", self.goal_callback, 10
        )

        # path publisher (default QoS is fine)
        self.path_pub = self.create_publisher(Path, "/dijkstra/path", 10)

        # visited_map publisher: use TRANSIENT_LOCAL so subscribers expecting latched map get it
        visited_qos = QoSProfile(depth=10)
        visited_qos.durability = DurabilityPolicy.TRANSIENT_LOCAL
        self.map_pub = self.create_publisher(OccupancyGrid, "/dijkstra/visited_map", visited_qos)

        # robot frame (configurable)
        self.declare_parameter("robot_frame", "base_footprint")
        self.robot_frame = self.get_parameter("robot_frame").get_parameter_value().string_value

        self.map_ = None
        self.visited_map_ = OccupancyGrid()

    def map_callback(self, map_msg: OccupancyGrid):
        self.map_ = map_msg
        self.visited_map_.header.frame_id = map_msg.header.frame_id
        self.visited_map_.info = map_msg.info
        self.visited_map_.data = [-1] * (map_msg.info.height * map_msg.info.width)

    def goal_callback(self, pose: PoseStamped):
        if self.map_ is None:
            self.get_logger().error("No map received!")
            return

        # reset visited map
        self.visited_map_.data = [-1] * (self.visited_map_.info.height * self.visited_map_.info.width)

        # Try to get robot pose in map frame. Try configured frame and fallbacks.
        frames_to_try = [self.robot_frame, "base_link", "base_footprint", "base"]
        map_to_base_tf = None
        for f in frames_to_try:
            try:
                # wait briefly for transform to appear (non-blocking polling)
                timeout_s = 1.0
                t0 = time.time()
                while time.time() - t0 < timeout_s:
                    try:
                        map_to_base_tf = self.tf_buffer.lookup_transform(self.map_.header.frame_id, f, Time())
                        break
                    except Exception:
                        time.sleep(0.05)
                if map_to_base_tf:
                    used_frame = f
                    break
            except Exception:
                # keep trying other frames
                continue

        if map_to_base_tf is None:
            self.get_logger().error(
                "Could not transform from map to any robot frame. "
                "Checked frames: %s. Inspect TF tree (ros2 topic echo /tf) and ensure map frame '%s' exists and the robot frame is published."
                % (frames_to_try, self.map_.header.frame_id)
            )
            return

        map_to_base_pose = Pose()
        map_to_base_pose.position.x = map_to_base_tf.transform.translation.x
        map_to_base_pose.position.y = map_to_base_tf.transform.translation.y
        map_to_base_pose.orientation = map_to_base_tf.transform.rotation

        path = self.plan(map_to_base_pose, pose.pose)

        if path.poses:
            self.get_logger().info("Shortest path found!")
            path.header.stamp = self.get_clock().now().to_msg()
            self.path_pub.publish(path)
        else:
            self.get_logger().warning("No path found to the goal.")

        # publish visited map once after planning (less spam)
        self.visited_map_.header.stamp = self.get_clock().now().to_msg()
        self.map_pub.publish(self.visited_map_)

    def plan(self, start: Pose, goal: Pose):
        explore_directions = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        pending_nodes = PriorityQueue()
        visited_nodes = set()

        start_node = self.world_to_grid(start)
        goal_node = self.world_to_grid(goal)

        start_node.cost = 0
        start_node.prev = None
        pending_nodes.put(start_node)
        visited_nodes.add(start_node)

        active_node = None

        while not pending_nodes.empty() and rclpy.ok():
            active_node = pending_nodes.get()

            # Goal found!
            if active_node == goal_node:
                break

            for dir_x, dir_y in explore_directions:
                new_node = active_node + (dir_x, dir_y)

                if self.pose_on_map(new_node):
                    idx = self.pose_to_cell(new_node)
                    if not (0 <= idx < len(self.map_.data)):
                        continue

                    if new_node not in visited_nodes and self.map_.data[idx] == 0:
                        new_node.cost = active_node.cost + 1
                        new_node.prev = active_node
                        pending_nodes.put(new_node)
                        visited_nodes.add(new_node)

            try:
                self.visited_map_.data[self.pose_to_cell(active_node)] = 10
            except Exception:
                pass

        path = Path()
        path.header.frame_id = self.map_.header.frame_id
        path.poses = []

        if active_node is None:
            return path

        if not (active_node == goal_node):
            return path

        while active_node and active_node.prev and rclpy.ok():
            last_pose = self.grid_to_world(active_node)
            last_pose_stamped = PoseStamped()
            last_pose_stamped.header.frame_id = self.map_.header.frame_id
            last_pose_stamped.header.stamp = self.get_clock().now().to_msg()
            last_pose_stamped.pose = last_pose
            path.poses.append(last_pose_stamped)
            active_node = active_node.prev

        path.poses.reverse()
        return path

    def pose_on_map(self, node: GraphNode):
        return 0 <= node.x < self.map_.info.width and 0 <= node.y < self.map_.info.height

    def world_to_grid(self, pose: Pose) -> GraphNode:
        grid_x = int((pose.position.x - self.map_.info.origin.position.x) / self.map_.info.resolution)
        grid_y = int((pose.position.y - self.map_.info.origin.position.y) / self.map_.info.resolution)
        return GraphNode(grid_x, grid_y)

    def grid_to_world(self, node: GraphNode) -> Pose:
        pose = Pose()
        # center of cell
        pose.position.x = node.x * self.map_.info.resolution + self.map_.info.origin.position.x + self.map_.info.resolution / 2.0
        pose.position.y = node.y * self.map_.info.resolution + self.map_.info.origin.position.y + self.map_.info.resolution / 2.0
        return pose

    def pose_to_cell(self, node: GraphNode):
        return node.y * self.map_.info.width + node.x


def main(args=None):
    rclpy.init(args=args)
    node = DijkstraPlanner()
    rclpy.spin(node)
    rclpy.shutdown()

if __name__ == '__main__':
    main()