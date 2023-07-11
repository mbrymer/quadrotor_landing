/*
    AER 1810 Quadrotor Landing Project
    Relative Pose Filter Node Main
*/

#include "relative_pose_filter_node.h"

int main(int argc, char **argv)
{
    ros::init(argc,argv,"relative_pose_filter");
    ros::NodeHandle filter_nh("~");

    RelativePoseFilterNode relative_pose_filter(filter_nh);

    // ros::AsyncSpinner spinner(2);
    // ros::waitForShutdown();
    ros::spin();

    return 0;
}