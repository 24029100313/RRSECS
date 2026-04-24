#!/bin/bash

#read -p "Please input filepath: " local_dir

local_dir="$1"

remote_port="12321"
remote_user="root"
remote_server="192.168.29.226"
remote_dir="/data/lxq"

remote_passwd="   "


sshpass -p "$remote_passwd" scp -r -P "$remote_port" "$local_dir" "$remote_user@$remote_server:$remote_dir"