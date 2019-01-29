echo "mkdir -p ~/ros_scripts" >&2
ssh ubuntu@car10 mkdir -p ros_scripts
echo "scp'ing..." >&2
scp -r . ubuntu@car10:ros_scripts/