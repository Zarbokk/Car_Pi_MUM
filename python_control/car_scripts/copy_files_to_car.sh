echo "mkdir -p ~/ros_scripts" >&2
ssh ubuntu@$rospi mkdir -p ~/ros_scripts
echo "scp'ing..." >&2
scp -r . ubuntu@$rospi:~/ros_scripts/
