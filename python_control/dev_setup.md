Add the line `10.42.0.1  rospi` as alias to `/etc/hosts` file.

Follow [this guide](https://serverfault.com/questions/241588/how-to-automate-ssh-login-with-password) to not need to use password to login. If you already have SSH Keys, simply type `ssh-copy-id ubuntu@rospi`. Now you can login via `ssh ubuntu@rospi`.


Connect to WIFI.
On the car, add your IP you got (check with `ifconfig`) to `/etc/hosts` with alias. add that alias to `.bashrc`, for example via `export ROS_MASTER_URI=http://nalbers-ubuntu:11311`.
