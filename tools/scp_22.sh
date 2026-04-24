#!/bin/bash

#read -p "Please input filepath: " local_dir

local_dir="$1"

remote_port="23432"
remote_user="root"
remote_server="192.168.29.122"
remote_dir="/root/lxq/FMs/InternVL2_5"

remote_passwd="0000"


sshpass -p "$remote_passwd" scp -r -P "$remote_port" "$local_dir" "$remote_user@$remote_server:$remote_dir"