/*
    AER 1810 Quadrotor Landing Project
    Relative Pose EKF Node Main
*/

#include "relative_pose_EKF_node.hpp"

int main(int argc, char **argv)
{
    ros::init(argc,argv,"relative_pose_EKF");
    ros::NodeHandle EKF_nh;

    RelativePoseEKFNode rel_pose_EKF(EKF_nh);

    // ros::AsyncSpinner spinner(2);
    // ros::waitForShutdown();
    ros::spin();

    return 0;
}