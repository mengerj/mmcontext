#!/bin/bash

# Remote Jupter Notebook server
# -----------------------------------------
# This script tries to run a jupyter notebook server on a remote machine,
# reached with a proxy jump and forwards the port, so the notebook server
# can be opened in the local browser.
# If you don't have SSH keys deposited for logging in, this script will ask
# you three times for your password. So setting up keys saves you time :)

# The path to the virtual environment folder on the GPU server
REMOTE_VENV_PATH="../.venv"

# Open this folder in the notebook server
NOTEBOOKS_PATH="~/"

# Assuming the current user is available on the proxy and remote machine
# If this differs, please enter your username for PROXY_USER and REMOTE_USER
#PROXY_USER=$(id --user --name)
# For me this returns the correct user (might not be the case for everyone. Check in the shell with id -F
# use can also hard code your name that you need for the cluster login
PROXY_USER=$(id -F)

PROXY_HOST="biom5.imbi.uni-freiburg.de"

#REMOTE_USER=$(id --user --name)
REMOTE_USER=$(id -F)
REMOTE_HOST="imbis236l40"

LOCAL_PORT=8080

cleanup() {
    echo "Stopping remote notebook server"
}
trap cleanup EXIT

# Check if the remote virtual environment path exists
if ! ssh -J "$PROXY_USER@$PROXY_HOST" "$REMOTE_USER@$REMOTE_HOST" "[ -d $REMOTE_VENV_PATH ]"; then
    echo "The virtual environment path '$REMOTE_VENV_PATH' does not exist on the remote machine."
    echo "If you already created one, please set the REMOTE_VENV_PATH in this script accordingly. If none exists, create it using the following command on the remote machine:"
    echo "  python3 -m venv $REMOTE_VENV_PATH"
    echo "Then install Jupyter Notebook in the virtual environment:"
    echo "  source $REMOTE_VENV_PATH/bin/activate && pip install notebook"
    exit 1
fi

# Find free port on the remote machine for the jupyter server, to avoid
# port conflicts between cluster users
echo "Finding free port on remote machine"
REMOTE_PORT=$(ssh -J "$PROXY_USER@$PROXY_HOST" "$REMOTE_USER@$REMOTE_HOST" \
    "python3 -c 'import socket; s=socket.socket(); s.bind((\"\", 0)); print(s.getsockname()[1]); s.close()'")

# Create the SSH tunnel and run Jupyter Notebook on the remote machine
echo "Establishing SSH tunnel"
ssh -L "$LOCAL_PORT:localhost:$REMOTE_PORT" -J "$PROXY_USER@$PROXY_HOST" "$REMOTE_USER@$REMOTE_HOST" \
    "bash --login -c 'source $REMOTE_VENV_PATH/bin/activate && \
    echo Starting Jupyter Notebook on port $REMOTE_PORT && \
    jupyter notebook --no-browser --port=$REMOTE_PORT'" --notebook-dir=$NOTEBOOKS_PATH 2>&1 | while read line; do

    # The first time we find a link with the token, we replace it with
    # a link containing the local port of the tunnel and try to open it
    # in the browser.
    if [[ $line =~ /tree\?token=([a-zA-Z0-9]+) ]]; then
        if [[ -z "$TOKEN" ]]; then
            TOKEN="${BASH_REMATCH[1]}"
            URL="http://localhost:$LOCAL_PORT/tree?token=$TOKEN"
            echo "Remote Jupyter Notebook is running! Open your browser and go to:"
            echo $URL

            # Automatically open the browser
            if command -v xdg-open >/dev/null; then
                xdg-open "$URL"  # Linux
            elif command -v open >/dev/null; then
                open "$URL"  # macOS
            elif command -v start >/dev/null; then
                start "$URL"  # Windows
            else
                echo "No suitable command found to open the browser. Please open the link manually."
            fi
        fi
    else
        # Skip other lines with misleading instructions
        if [[ $line != *"To access the server"* &&
              $line != *"file://"* &&
              $line != *"Or copy and paste"* ]]; then
          echo "$line"
        fi
    fi

done
