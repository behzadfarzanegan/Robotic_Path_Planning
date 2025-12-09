#include "bumperbot_planning/a_star_planner.hpp"
#include <rclcpp/qos.hpp>
#include <rmw/qos_profiles.h>
#include "geometry_msgs/msg/transform_stamped.hpp"
#include <queue>
#include <algorithm> // Required for std::reverse

namespace bumperbot_planning
{
AStarPlanner::AStarPlanner(): Node("a_star_node")
{
    tf_buffer_ = std::make_unique<tf2_ros::Buffer>(get_clock());
    tf_listener_ = std::make_shared<tf2_ros::TransformListener>(*tf_buffer_);

    rclcpp::QoS map_qos(10);
    map_qos.durability(RMW_QOS_POLICY_DURABILITY_TRANSIENT_LOCAL);
    map_sub_ = create_subscription<nav_msgs::msg::OccupancyGrid>(
        "/map", map_qos, std::bind(&AStarPlanner::mapCallback, this, std::placeholders::_1));  
    pose_sub_ = create_subscription<geometry_msgs::msg::PoseStamped>(
        "/goal_pose", 10, std::bind(&AStarPlanner::poseCallback, this, std::placeholders::_1));
    path_pub_ = create_publisher<nav_msgs::msg::Path>("/a_star/path", 10);
    map_pub_ = create_publisher<nav_msgs::msg::OccupancyGrid>("/a_star/visited_map", 10);

}

    void AStarPlanner::mapCallback(const nav_msgs::msg::OccupancyGrid::SharedPtr map)
    {
        map_ = map;
        visited_map_.header.frame_id = map->header.frame_id;
        visited_map_.info = map->info;
        visited_map_.data = std::vector<int8_t>(visited_map_.info.height * visited_map_.info.width, -1);
    }

    void AStarPlanner::poseCallback(const geometry_msgs::msg::PoseStamped::SharedPtr pose)
    {
        if(!map_){
            RCLCPP_ERROR(get_logger(), "No map received yet");
            return;
        }
        // Reset the visited map and path for each new goal
        visited_map_.data = std::vector<int8_t>(visited_map_.info.height*visited_map_.info.width, -1);
        nav_msgs::msg::Path empty_path;
        path_pub_->publish(empty_path);
        
        geometry_msgs::msg::TransformStamped map_to_base_tf;
        try{
            map_to_base_tf = tf_buffer_->lookupTransform(map_->header.frame_id, "base_link", tf2::TimePointZero);
        }catch(const tf2::TransformException & ex){
            RCLCPP_ERROR(get_logger(), "Could not transform from %s to %s: %s", map_->header.frame_id.c_str(), "base_link", ex.what());
            return;
        }
        geometry_msgs::msg::Pose start_pose;
        start_pose.position.x = map_to_base_tf.transform.translation.x;
        start_pose.position.y = map_to_base_tf.transform.translation.y;
        start_pose.orientation = map_to_base_tf.transform.rotation;
        
        auto path = plan(start_pose, pose->pose);
        
        if (!path.poses.empty())
        {
            RCLCPP_INFO(get_logger(), "Publishing path with %zu poses", path.poses.size());
            path_pub_->publish(path);
            map_pub_->publish(visited_map_); // <--- FIX 1: Moved Publisher HERE (outside loop)
        }else{
            RCLCPP_WARN(get_logger(), "No path found");
        }
    } 
    
    nav_msgs::msg::Path AStarPlanner::plan(const geometry_msgs::msg::Pose & start, const geometry_msgs::msg::Pose & goal)
    {
        GraphNode start_node = worldToGrid(start);
        GraphNode goal_node = worldToGrid(goal);

        // --- 1. Check if Goal is Out of Bounds ---
        if(!poseOnMap(goal_node) || !poseOnMap(start_node)){
            RCLCPP_WARN(get_logger(), "Start or Goal pose is out of map bounds!");
            return nav_msgs::msg::Path();
        }

        // --- 2. Check if Goal is Inside a Wall ---
        unsigned int goal_idx = poseToCell(goal_node);
        int8_t map_value = map_->data.at(goal_idx);

            // 0 = Free, 100 = Occupied, -1 = Unknown
        if (map_value != 0) 
            {
                RCLCPP_WARN(get_logger(), "Goal is invalid! Map value at goal is: %d", map_value);
                return nav_msgs::msg::Path();
            }
        std::priority_queue< GraphNode, std::vector<GraphNode>, std::greater<GraphNode>> pending_node;
        // Removed visited_nodes vector (redundant and slow)
        std::vector<std::pair<int,int>> explore_directions = {{1,0}, {-1,0}, {0,1}, {0,-1}};


        start_node.heuristic = manhattanDistance(start_node, goal_node);
        pending_node.push(worldToGrid(start));
        GraphNode active_node;
        
        while(!pending_node.empty() && rclcpp::ok()){
            active_node = pending_node.top();
            pending_node.pop();
            
            // Optimization: Skip if already closed
            if (visited_map_.data.at(poseToCell(active_node)) == -106) continue;
            
            // Mark as Closed
            visited_map_.data.at(poseToCell(active_node)) = -106;

            if(active_node == worldToGrid(goal)){
                RCLCPP_INFO(get_logger(), "Goal reached at (%d, %d) with cost %d", active_node.x, active_node.y, active_node.cost);
                break;
            }
            
            for(const auto & dir: explore_directions){
                GraphNode new_node = active_node + dir;
                
                if(poseOnMap(new_node) 
                   && visited_map_.data.at(poseToCell(new_node)) != -106
                   && map_->data.at(poseToCell(new_node)) == 0)
                {
                    new_node.cost = active_node.cost +1;
                    new_node.heuristic = manhattanDistance(new_node, goal_node);
                    new_node.prev = std::make_shared<GraphNode>(active_node);
                    pending_node.push(new_node);
                    
                    // Mark as Open/Seen
                    visited_map_.data.at(poseToCell(new_node)) = 50; 
                }
            }
            map_pub_->publish(visited_map_);
        }

        nav_msgs::msg::Path path;
        path.header.frame_id = map_->header.frame_id;
        
        GraphNode node_iter = active_node; 
        while (node_iter.prev && rclcpp::ok())
        {
            geometry_msgs::msg::Pose last_pose = gridToWorld(node_iter);
            geometry_msgs::msg::PoseStamped last_pose_stamped;
            last_pose_stamped.header.frame_id = map_->header.frame_id;
            last_pose_stamped.pose = last_pose;
            path.poses.push_back(last_pose_stamped);
            node_iter = *node_iter.prev;
        }
        
        // <---  Add the Start Node (which has no prev)
        if (!path.poses.empty() || active_node == worldToGrid(start)) {
             geometry_msgs::msg::Pose start_pose = gridToWorld(node_iter);
             geometry_msgs::msg::PoseStamped start_stamped;
             start_stamped.header.frame_id = map_->header.frame_id;
             start_stamped.pose = start_pose;
             path.poses.push_back(start_stamped);
        }
        
        std::reverse(path.poses.begin(), path.poses.end());
        return path;

    }

    GraphNode AStarPlanner::worldToGrid(const geometry_msgs::msg::Pose & pose)
    {
        int grid_x = static_cast<int>((pose.position.x - map_->info.origin.position.x)/ map_->info.resolution);
        int grid_y = static_cast<int>((pose.position.y - map_->info.origin.position.y)/ map_->info.resolution);
        return GraphNode(grid_x, grid_y);
    }

    bool AStarPlanner::poseOnMap(const GraphNode & node)
    {
        return(node.x >= 0 && node.x < static_cast<int>(map_->info.width) &&
               node.y >= 0 && node.y < static_cast<int>(map_->info.height));   
    }

    unsigned int AStarPlanner::poseToCell( const GraphNode & node)
    {
        return node.y * map_->info.width + node.x;
    }

    geometry_msgs::msg::Pose AStarPlanner::gridToWorld(const GraphNode & node)
    {
        geometry_msgs::msg::Pose pose;
        pose.position.x = node.x * map_->info.resolution + map_->info.origin.position.x;
        pose.position.y = node.y * map_->info.resolution + map_->info.origin.position.y;
        return pose;
    }

    double AStarPlanner::manhattanDistance(const GraphNode & node, const GraphNode & goal_node)
    {
        return std::abs(node.x - goal_node.x) + std::abs(node.y - goal_node.y);
    }



}

int main(int argc, char ** argv)
{
    rclcpp::init(argc,argv);
    auto planner_node = std::make_shared<bumperbot_planning::AStarPlanner>();
    rclcpp::spin(planner_node);
    rclcpp::shutdown();
    return 0;
}