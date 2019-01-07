echo "mkdir -p ~/ros_scripts" >&2
ssh ubuntu@ubiquityrobot mkdir -p ros_scripts
echo "scp'ing..." >&2
scp -r . ubuntu@ubiquityrobot:ros_scripts/

ssh ubuntu@ubiquityrobot "./ros_scripts/start_notes.sh"