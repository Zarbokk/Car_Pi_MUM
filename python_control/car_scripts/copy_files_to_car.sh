echo "mkdir -p ~/ros_scripts" >&2
ssh ubuntu@car03 mkdir -p ros_scripts
echo "scp'ing..." >&2
scp -r . ubuntu@car03:ros_scripts/