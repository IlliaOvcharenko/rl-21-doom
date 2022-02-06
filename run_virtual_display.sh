# Run with source
# Run virtual XVBF  display and start VNC to braodcast it outside docker container
# Use any VNC viewer and port 5900 to connect

sudo  Xvfb "$DISPLAY" -screen 0 1024x768x24 &
x11vnc -display "$DISPLAY" -bg -nopw -repeat -listen "$(hostname)" -xkb

