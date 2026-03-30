/**
 * @file RLControlNewPlugin.cpp
 * @brief ROS2版本的humanoid rl plugin
 * @version 2.0
 * @date 2025-09-17
 *
 * @修改记录1: zyj 2025-09-17 适配天工dex
 *
 */

#include "RLControlNewPlugin.h"

#include <pluginlib/class_list_macros.hpp>
#include <rclcpp_components/register_node_macro.hpp>
#include <stdio.h>
#include <thread>
#include <chrono>
#include <cmath>
#include <iostream>
#include <time.h>
#include <fstream>
#include "broccoli/core/Time.hpp"
#include "spdlog/async.h"
#include "spdlog/sinks/basic_file_sink.h"
#include "spdlog/sinks/rotating_file_sink.h"
#include "spdlog/sinks/stdout_color_sinks.h"
#include <ament_index_cpp/get_package_share_directory.hpp>

#include "funcSPTrans.h" // 串并联转换

using namespace broccoli::core;

namespace rl_control_new
{

    // 带 NodeOptions 的构造函数（用于可组合节点）
    RLControlNewPlugin::RLControlNewPlugin(const rclcpp::NodeOptions &options)
        : rclcpp::Node("rl_control_new_plugin", options)
    {
        onInit();
    }

    bool RLControlNewPlugin::LoadConfig(const std::string &_config_file)
    {
        config_ = YAML::LoadFile(_config_file);
        if (!config_)
        {
            RCLCPP_ERROR(this->get_logger(), "Failed to load config file: %s", _config_file.c_str());
            return false;
        }
        action_num = config_["actions_size"].as<int>();
        motor_num = config_["motor_num"].as<int>();
        simulation = config_["simulation"].as<bool>();

        dt = config_["dt"].as<double>();
        ct_scale = Eigen::Map<Eigen::VectorXd>(config_["ct_scale"].as<std::vector<double>>().data(), motor_num);
        robot_data.config_file_ = _config_file;
        robot_data.joint_kp_p_ = Eigen::Map<Eigen::VectorXd>(config_["joint_kp_p"].as<std::vector<double>>().data(), motor_num);
        robot_data.joint_kd_p_ = Eigen::Map<Eigen::VectorXd>(config_["joint_kd_p"].as<std::vector<double>>().data(), motor_num);

        return true;
    }

    void RLControlNewPlugin::onInit()
    {
        std::string pkg_path = ament_index_cpp::get_package_share_directory("rl_control_new");
        _config_file = pkg_path + "/config/tg22_config.yaml";
        if (!LoadConfig(_config_file))
        {
            std::cout << "load config file error: " << _config_file << std::endl;
        }

        idMap.bodyCanIdMapInit();
        ////// 腿12 + 手臂8 + 浮动基6 = 26 //////
        int whole_joint_num = 26;
        pos_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        vel_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        tau_fed_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        pos_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        vel_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        tau_cmd_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        ct_scale_midVec = Eigen::VectorXd::Zero(whole_joint_num);
        temperature_midVec = Eigen::VectorXd::Zero(whole_joint_num);

        pubLegMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/leg/cmd_ctrl", 100);
        pubArmMotorCmd = this->create_publisher<bodyctrl_msgs::msg::CmdMotorCtrl>("/arm/cmd_ctrl", 100);
        // 腰部和头部归零
        waists_cmd_pub_ = this->create_publisher<bodyctrl_msgs::msg::CmdSetMotorPosition>("/waist/cmd_pos", 1);

        subLegState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
            "/leg/status", 100,
            std::bind(&RLControlNewPlugin::LegMotorStatusMsg, this, std::placeholders::_1));
        subArmState = this->create_subscription<bodyctrl_msgs::msg::MotorStatusMsg>(
            "/arm/status", 100,
            std::bind(&RLControlNewPlugin::ArmMotorStatusMsg, this, std::placeholders::_1));
        subImuXsens = this->create_subscription<bodyctrl_msgs::msg::Imu>(
            "/imu/status", 100,
            std::bind(&RLControlNewPlugin::OnXsensImuStatusMsg, this, std::placeholders::_1));
        subJoyCmd = this->create_subscription<sensor_msgs::msg::Joy>(
            "/sbus_data", 100,
            std::bind(&RLControlNewPlugin::xbox_map_read, this, std::placeholders::_1));

        // Initialize member variables
        funS2P = new funcSPTrans;

        pi = 3.14159265358979;
        rpm2rps = pi / 30.0;
        Q_a = Eigen::VectorXd::Zero(motor_num);
        Qdot_a = Eigen::VectorXd::Zero(motor_num);
        Tor_a = Eigen::VectorXd::Zero(motor_num);
        Q_d = Eigen::VectorXd::Zero(motor_num);
        Qdot_d = Eigen::VectorXd::Zero(motor_num);
        Tor_d = Eigen::VectorXd::Zero(motor_num);
        Temperature = Eigen::VectorXd::Zero(motor_num);

        q_a = Eigen::VectorXd::Zero(motor_num);
        qdot_a = Eigen::VectorXd::Zero(motor_num);
        tor_a = Eigen::VectorXd::Zero(motor_num);
        q_d = Eigen::VectorXd::Zero(motor_num);
        qdot_d = Eigen::VectorXd::Zero(motor_num);
        tor_d = Eigen::VectorXd::Zero(motor_num);

        q_a_p = Eigen::VectorXd::Zero(4);
        qdot_a_p = Eigen::VectorXd::Zero(4);
        tor_a_p = Eigen::VectorXd::Zero(4);
        q_d_p = Eigen::VectorXd::Zero(4);
        qdot_d_p = Eigen::VectorXd::Zero(4);
        tor_d_p = Eigen::VectorXd::Zero(4);

        q_a_s = Eigen::VectorXd::Zero(4);
        qdot_a_s = Eigen::VectorXd::Zero(4);
        tor_a_s = Eigen::VectorXd::Zero(4);
        q_d_s = Eigen::VectorXd::Zero(4);
        qdot_d_s = Eigen::VectorXd::Zero(4);
        tor_d_s = Eigen::VectorXd::Zero(4);

        Q_a_last = Eigen::VectorXd::Zero(motor_num);
        Qdot_a_last = Eigen::VectorXd::Zero(motor_num);
        Tor_a_last = Eigen::VectorXd::Zero(motor_num);
        ct_scale_midVec.head(12) << ct_scale.head(12);

        zero_pos = Eigen::VectorXd::Zero(motor_num);
        std::stringstream ss;

        // 零位调整 - 用于电机原始位置与标准位置的转换
        if (config_["zero_pos_offset"])
        {
            for (int32_t i = 0; i < motor_num; ++i)
            {
                zero_pos[i] = config_["zero_pos_offset"][i].as<double>();
                ss << zero_pos[i] << "  ";
            }
        }

        if (config_["xsense_data_roll_offset"])
        {
            // 日志输出在调试时有用，但在正常运行中可以省略
        }

        init_pos = Eigen::VectorXd::Zero(motor_num);
        motor_dir = Eigen::VectorXd::Ones(motor_num);
        zero_cnt = Eigen::VectorXd::Zero(motor_num);
        zero_offset = Eigen::VectorXd::Zero(motor_num);
        xsense_data = Eigen::VectorXd::Zero(13);
        data = Eigen::VectorXd::Zero(450);

        sleep(1);
        std::thread([this]()
                    { rlControl(); })
            .detach();
    }

    // 混合模式测试
    void RLControlNewPlugin::rlControl()
    {
        // set sched-strategy
        struct sched_param sched;
        int max_priority;

        max_priority = sched_get_priority_max(SCHED_RR);
        sched.sched_priority = max_priority;

        if (sched_setscheduler(gettid(), SCHED_RR, &sched) == -1)
        {
            printf("Set Scheduler Param, ERROR:%s\n", strerror(errno));
        }
        usleep(1000);
        // set sched-strategy

        // joystick init
        Joystick_humanoid joystick_humanoid;
        joystick_humanoid.init();

        // robot FSM init
        RobotFSM *robot_fsm = get_robot_FSM(robot_data);
        // robot_interface init
        RobotInterface *robot_interface = get_robot_interface();
        robot_interface->Init();

        long count = 0;
        double t_now = 0;

        Time start_time;
        Time period(0, dt * 1e9); // Convert dt to nanoseconds
        Time sleep2Time;
        Time timer;
        timespec sleep2Time_spec;
        double timeFSM = 0.0;
        Time timer1, timer2, timer3, total_time;

        while (queueLegMotorState.empty() || queueArmMotorState.empty())
        {
            RCLCPP_WARN(this->get_logger(), "[RobotFSM] queue is empty");
            std::this_thread::sleep_for(std::chrono::milliseconds(10)); // 睡眠10毫秒(0.01秒)
        }
        while (!queueLegMotorState.empty())
        {
            auto msg = queueLegMotorState.pop();
            for (auto &one : msg->status)
            {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }

        while (!queueArmMotorState.empty())
        {
            auto msg = queueArmMotorState.pop();
            for (auto &one : msg->status)
            {
                int index = idMap.getIndexById(one.name);
                pos_fed_midVec(index) = one.pos;
                vel_fed_midVec(index) = one.speed;
                tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                temperature_midVec(index) = one.temperature;
            }
        }

        Q_a.head(motor_num) << pos_fed_midVec.head(motor_num);
        Qdot_a.head(motor_num) << vel_fed_midVec.head(motor_num);
        Tor_a.head(motor_num) << tau_fed_midVec.head(motor_num);
        Temperature.head(motor_num) << temperature_midVec.head(motor_num);

        init_pos = Q_a;
        Q_a_last = Q_a;
        Qdot_a_last = Qdot_a;
        Tor_a_last = Tor_a;

        for (int i = 0; i < motor_num; i++)
        {
            q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) + zero_offset(i);
            zero_cnt(i) = (q_a(i) > pi) ? -1.0 : zero_cnt(i);
            zero_cnt(i) = (q_a(i) < -pi) ? 1.0 : zero_cnt(i);
            q_a(i) += zero_cnt(i) * 2.0 * pi;
        }

        bool is_disable{false};
        int cnt = 0;
        xbox_flag flag_;
        flag_.is_disable = 0;
        while (1)
        {
            total_time = timer.currentTime() - start_time;
            start_time = timer.currentTime();

            while (!queueLegMotorState.empty())
            {
                auto msg = queueLegMotorState.pop();
                for (auto &one : msg->status)
                {
                    int index = idMap.getIndexById(one.name);
                    pos_fed_midVec(index) = one.pos;
                    vel_fed_midVec(index) = one.speed;
                    tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                    temperature_midVec(index) = one.temperature;
                }
            }

            while (!queueArmMotorState.empty())
            {
                auto msg = queueArmMotorState.pop();
                for (auto &one : msg->status)
                {
                    int index = idMap.getIndexById(one.name);
                    pos_fed_midVec(index) = one.pos;
                    vel_fed_midVec(index) = one.speed;
                    tau_fed_midVec(index) = one.current * ct_scale_midVec(index);
                    temperature_midVec(index) = one.temperature;
                }
            }

            Q_a.head(motor_num) << pos_fed_midVec.head(motor_num);
            Qdot_a.head(motor_num) << vel_fed_midVec.head(motor_num);
            Tor_a.head(motor_num) << tau_fed_midVec.head(motor_num);
            Temperature.head(motor_num) << temperature_midVec.head(motor_num);

            while (!queueImuXsens.empty())
            {
                auto msg = queueImuXsens.pop();
                // set xsens imu buf
                xsense_data(0) = msg->euler.yaw;
                xsense_data(1) = msg->euler.pitch;
                xsense_data(2) = msg->euler.roll;
                xsense_data(3) = msg->angular_velocity.x;
                xsense_data(4) = msg->angular_velocity.y;
                xsense_data(5) = msg->angular_velocity.z;
                xsense_data(6) = msg->linear_acceleration.x;
                xsense_data(7) = msg->linear_acceleration.y;
                xsense_data(8) = msg->linear_acceleration.z;
            }

            double pitch = xsense_data(1);
            double roll = xsense_data(2);
            Eigen::Vector3d gyro_vec = xsense_data.segment<3>(3);
            double gyro_norm = gyro_vec.norm();

            if (std::abs(pitch) >= 0.8 || std::abs(roll) >= 0.8 ||
                gyro_norm > 5.0)
            {
                // 日志输出在调试时有用，但在正常运行中可以省略
                // std::cout << "[IMU] Pitch/Roll/AngularVel 超限，进入STOP" << std::endl;
            }

            while (!queueJoyCmd.empty())
            {
                auto msg = queueJoyCmd.pop();
                // set joy cmd buf
                if (msg->axes.size() == 12)
                {
                    // 云卓
                    xbox_map.a = msg->axes[8];
                    xbox_map.b = msg->axes[9];
                    xbox_map.c = msg->axes[10];
                    xbox_map.d = msg->axes[11];
                    xbox_map.e = msg->axes[4];
                    xbox_map.f = msg->axes[7];
                    xbox_map.g = msg->axes[5];
                    xbox_map.h = msg->axes[6];
                    xbox_map.x1 = msg->axes[3];
                    xbox_map.x2 = msg->axes[1];
                    xbox_map.y1 = msg->axes[2];
                    xbox_map.y2 = msg->axes[0];
                }
                else if (msg->axes.size() == 8)
                {
                    // xbox }
                    //  {x,y, yaw} command
                    xbox_map.x1 = msg->axes[0]; // lx
                    xbox_map.x2 = msg->axes[3]; // rx
                    xbox_map.y1 = msg->axes[1]; // ly
                    xbox_map.y2 = msg->axes[4]; // ry
                    //
                    xbox_map.a = msg->buttons[0]; // a
                    xbox_map.b = msg->buttons[1]; // b
                    xbox_map.c = msg->buttons[3]; // y
                    xbox_map.d = msg->buttons[2]; // x
                    xbox_map.e = msg->buttons[4]; // lb
                    xbox_map.f = msg->buttons[5]; // rb
                    xbox_map.g = msg->buttons[6]; // select
                    xbox_map.h = msg->buttons[7]; // start
                }
            }

            // calculate Command
            t_now = count * dt;
            robot_data.time_now_ = t_now;

            for (int i = 0; i < motor_num; i++)
            {
                if (fabs(Q_a(i) - Q_a_last(i)) > pi)
                {
                    // 错误处理：关节角度跳变过大
                    Q_a(i) = Q_a_last(i);
                    Qdot_a(i) = Qdot_a_last(i);
                    Tor_a(i) = Tor_a_last(i);
                }
            }
            Q_a_last = Q_a;
            Qdot_a_last = Qdot_a;
            Tor_a_last = Tor_a;

            // real feedback
            for (int i = 0; i < motor_num; i++)
            {
                q_a(i) = (Q_a(i) - zero_pos(i)) * motor_dir(i) + zero_offset(i);
                q_a(i) += zero_cnt(i) * 2.0 * pi;
                qdot_a(i) = Qdot_a(i) * motor_dir(i);
                tor_a(i) = Tor_a(i) * motor_dir(i);
            }

            if (!simulation)
            {
                // 并转串
                // 提取左右脚两个踝关节
                q_a_p << q_a.segment(4, 2), q_a.segment(10, 2);
                qdot_a_p << qdot_a.segment(4, 2), qdot_a.segment(10, 2);
                tor_a_p << tor_a.segment(4, 2), tor_a.segment(10, 2);

                // 计算并转串
                funS2P->setPEst(q_a_p, qdot_a_p, tor_a_p);
                funS2P->calcFK();
                funS2P->calcIK();
                // 获取结果
                funS2P->getSState(q_a_s, qdot_a_s, tor_a_s);

                // 结果覆盖
                q_a.segment(4, 2) = q_a_s.head(2);
                q_a.segment(10, 2) = q_a_s.tail(2);
                qdot_a.segment(4, 2) = qdot_a_s.head(2);
                qdot_a.segment(10, 2) = qdot_a_s.tail(2);
                tor_a.segment(4, 2) = tor_a_s.head(2);
                tor_a.segment(10, 2) = tor_a_s.tail(2);
            }

            // add offset
            if (config_["xsense_data_roll_offset"])
            {
                double offset = config_["xsense_data_roll_offset"].as<double>();
                xsense_data(2) += offset;
            }

            // get state（whole_joint_num个从后取motor_num个）
            robot_data.q_a_.tail(motor_num) = q_a;
            robot_data.q_dot_a_.tail(motor_num) = qdot_a;
            robot_data.tau_a_.tail(motor_num) = tor_a;
            robot_data.imu_data_ = xsense_data.head(9); // xsense imu

            // get state
            robot_interface->GetState(timeFSM, robot_data);

            joystick_humanoid.xbox_flag_update(xbox_map);

            xbox_flag flag_ = joystick_humanoid.get_xbox_flag();
            printXboxFlag(flag_);
            timer1 = timer.currentTime() - start_time;

            if (robot_fsm->getCurrentState() == FSMStateName::STOP && flag_.fsm_state_command == "gotoZero")
            {
                // 腰部回零
                bodyctrl_msgs::msg::CmdSetMotorPosition waist_msg;
                for (int i = 0; i < 1; i++)
                {
                    bodyctrl_msgs::msg::SetMotorPosition cmd;
                    cmd.name = 31;
                    cmd.pos = 0.0;
                    cmd.spd = 0.3;
                    cmd.cur = 3;
                    waist_msg.header.stamp = rclcpp::Clock().now();

                    waist_msg.cmds.push_back(cmd);
                }
                waists_cmd_pub_->publish(waist_msg);
            }

            // rl fsm
            robot_fsm->RunFSM(flag_);

            timer2 = timer.currentTime() - start_time - timer1;

            // set command
            robot_interface->SetCommand(robot_data);

            timeFSM += dt;

            q_d = robot_data.q_d_.tail(motor_num);
            qdot_d = robot_data.q_dot_d_.tail(motor_num);
            tor_d = robot_data.tau_d_.tail(motor_num);

            if (!simulation)
            {
                // 串转并
                // 取踝关节两关节
                q_d_s << q_d.segment(4, 2), q_d.segment(10, 2);
                qdot_d_s << qdot_d.segment(4, 2), qdot_d.segment(10, 2);
                tor_d_s << tor_d.segment(4, 2), tor_d.segment(10, 2);

                // 转换
                funS2P->setSDes(q_d_s, qdot_d_s, tor_d_s);
                funS2P->calcJointPosRef();
                funS2P->calcJointTorDes();
                funS2P->getPDes(q_d_p, qdot_d_p, tor_d_p);

                // 覆盖原来的值
                q_d.segment(4, 2) = q_d_p.head(2);
                q_d.segment(10, 2) = q_d_p.tail(2);
                qdot_d.segment(4, 2) = qdot_d_p.head(2);
                qdot_d.segment(10, 2) = qdot_d_p.tail(2);
                tor_d.segment(4, 2) = tor_d_p.head(2);
                tor_d.segment(10, 2) = tor_d_p.tail(2);
            }

            for (int i = 0; i < motor_num; i++)
            {
                Q_d(i) = (q_d(i) - zero_offset(i) - zero_cnt(i) * 2.0 * pi) * motor_dir(i) + zero_pos(i);
                Qdot_d(i) = qdot_d(i) * motor_dir(i);
                Tor_d(i) = tor_d(i) * motor_dir(i);
            }

            if (robot_fsm->disable_joints_)
            {
                robot_data.joint_kp_p_.setZero();
                robot_data.joint_kd_p_.setZero();
                Tor_d.setZero();
                is_disable = true;
            }

            pos_cmd_midVec.head(motor_num) << Q_d.head(motor_num);
            vel_cmd_midVec.head(motor_num) << Qdot_d.head(motor_num);
            tau_cmd_midVec.head(motor_num) << Tor_d.head(motor_num);

            // Send Command motorctrl mode
            bodyctrl_msgs::msg::CmdMotorCtrl leg_msg;
            leg_msg.header.stamp = this->get_clock()->now();

            for (int i = 0; i < 12; i++)
            {
                bodyctrl_msgs::msg::MotorCtrl cmd;
                cmd.name = idMap.getIdByIndex(i);
                cmd.kp = robot_data.joint_kp_p_(i);
                cmd.kd = robot_data.joint_kd_p_(i);
                cmd.pos = pos_cmd_midVec(i);
                cmd.spd = vel_cmd_midVec(i);
                cmd.tor = tau_cmd_midVec(i);
                leg_msg.cmds.push_back(cmd);
            }
            pubLegMotorCmd->publish(leg_msg);

            // Arm control
            bodyctrl_msgs::msg::CmdMotorCtrl arm_msg;
            arm_msg.header.stamp = this->get_clock()->now();

            // 发送手臂关节命令 (索引12-19，对应8个关节)
            std::vector<int> arm_indices = {12, 13, 14, 15, 16, 17, 18, 19};
            for (int index : arm_indices)
            {
                bodyctrl_msgs::msg::MotorCtrl cmd_arm;
                cmd_arm.name = idMap.getIdByIndex(index);
                cmd_arm.kp = robot_data.joint_kp_p_(index);
                cmd_arm.kd = robot_data.joint_kd_p_(index);
                cmd_arm.pos = pos_cmd_midVec(index);
                cmd_arm.spd = vel_cmd_midVec(index);
                cmd_arm.tor = tau_cmd_midVec(index);
                arm_msg.cmds.push_back(cmd_arm);
            }

            pubArmMotorCmd->publish(arm_msg);

            timer3 = timer.currentTime() - start_time - timer1 - timer2;
            sleep2Time = start_time + period;
            sleep2Time_spec = sleep2Time.toTimeSpec();
            clock_nanosleep(CLOCK_MONOTONIC, TIMER_ABSTIME, &(sleep2Time_spec), NULL);
            count++;

            if (is_disable)
            {
                break;
            }
        }
    }

    void RLControlNewPlugin::LegMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
    {
        auto wrapper = msg;
        queueLegMotorState.push(wrapper);
    }

    void RLControlNewPlugin::ArmMotorStatusMsg(const bodyctrl_msgs::msg::MotorStatusMsg::SharedPtr msg)
    {
        auto wrapper = msg;
        queueArmMotorState.push(wrapper);
    }

    void RLControlNewPlugin::OnXsensImuStatusMsg(const bodyctrl_msgs::msg::Imu::SharedPtr msg)
    {
        auto wrapper = msg;
        queueImuXsens.push(wrapper);
    }

    void RLControlNewPlugin::xbox_map_read(const sensor_msgs::msg::Joy::SharedPtr msg)
    {
        auto wrapper = msg;
        queueJoyCmd.push(wrapper);
    }

    void RLControlNewPlugin::printXboxFlag(const xbox_flag &flag)
    {
        std::cout << "======= Xbox Flag Info =======" << std::endl;
        std::cout << "fsm_state_command: " << flag.fsm_state_command << std::endl;
        std::cout << "is_disable: " << flag.is_disable << std::endl;
    }
} // namespace rl_control_new

// 注册可组合节点
RCLCPP_COMPONENTS_REGISTER_NODE(rl_control_new::RLControlNewPlugin)
