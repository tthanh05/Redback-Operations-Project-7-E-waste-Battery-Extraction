% Fake YOLO node – publishes a string on /detections
% Run this in MATLAB command window BEFORE starting Simulink

setenv('ROS_DOMAIN_ID', '50');

node = ros2node("/fake_yolo");
pub = ros2publisher(node, "/detections", "std_msgs/String");

msg = ros2message("std_msgs/String");
msg.data = 'battery_detected';

fprintf('Fake YOLO node started – sending "battery_detected" every 2 seconds\n');

while true
    send(pub, msg);
    fprintf('Sent: %s\n', msg.data);
    pause(2);
end