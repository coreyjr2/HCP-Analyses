# Use lsblk to check where the portable drive is in /dev
lsblk
# This can change when the system is rebooted

# Mount the device to an EXISTING directory
# The target directory should be in your home directory unless you want to share that path with other users.
# If you want to share the mounted fie directory, check with the sysadmin
# Example to mount to a device to a shared directory with wheel privilages
sudo mount /dev/sda2 /mnt/usb1

ls /mnt/usb1