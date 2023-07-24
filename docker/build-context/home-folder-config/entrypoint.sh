#!/bin/sh
set -e

# Check if the HOST_UID and HOST_GID are specified
if [ -n "${HOST_UID+x}" ] && [ -n "${HOST_GID+x}" ]; then
	# Change 'myuser' UID and GID to match the host user's UID and GID
	sudo usermod -u "$HOST_UID" "$(whoami)"
	sudo groupmod -g "$HOST_GID" "$(whoami)"
	# Ensure ownership of home directory is reset to 'myuser' after UID and GID change
	chown -R "$(whoami)":"$(whoami)" /home/"$(whoami)"
fi

# Switch to 'myuser'
su -c "$(whoami)"

# Run the main command with exec to replace the current process (bash) with the given command
# This is important because of how signals are handled (https://hynek.me/articles/docker-signals/)
exec "$@"
