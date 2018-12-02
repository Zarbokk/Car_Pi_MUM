| :warning: Alles hier ist derzeit noch Work in Progress.|
| --- |
# Python/ROS Ansteuerung
Im Folgenden wird die Ansteuerung des Autos durch [ROS (Roboter Operating System)](http://www.ros.org/) beschrieben, insbesondere:

1. Motor Steuerung
2. IMU auslesen
3. Kamera auslesen
4. Steuerung mit PS4 Controller
<!-- 
    TODO: überarbeiten
Dazu wird Folgendes benoetigt :
1. Linux/Mac
2. Python
3. ROS Kinetic
4. Eventuell andere Packages wie z.B. OpenCV
 -->
| :warning: Auf Windows/OS X ist die Installation von ROS äußerst schwierig. Für Ubuntu gibt es fertige Pakete für `apt`.|
| --- |

Das Image was auf die SD Karte gezogen werden muss ist hier downzuloaden: https://downloads.ubiquityrobotics.com/
Hier steht auch wie man sich mit dem PI verbindet.

| :heavy_exclamation_mark: |
-------------- | 
| - Bei Installation unbedingt auf automatisches Login achten, damit das WLAN automatisch startet|
| - Speichert das alte Image falls ihr zu Matlab zurueck wollt|


Fuer die ROS nutzung muss am anfang folgendes Erledigt werden:
Die Componenten (Auto + PC) muessen sich kennen. Dafuer sorgt die HOST Datei in /etc/hosts
Tutorial fuer die Netzwerksteuerung: http://wiki.ros.org/ROS/NetworkSetup

Als naechstes bevor die Scripte gestartet werden koennen muss roscore auf dem PI gestartet werden. Alle Daten die PI am ende ihrer Datei haben muessen auf dem PI gestartet werden. 

 
# Motor Steuerung
wird durch ROS_Listener_Motor_PI.py gestartet. Sendet alle empfangenen daten an die Motorsteuerung.
Das Script Motor_Steuerung_v.py kann die Geschwindigkeit einstellen als beispiel. Rest steht in der Datei selber. 
# Imu
Aktuell nur auf dem PI. Kann gelesen werden um Beschleunigung und Drehung festzustellen. IMU_ROS_NODE_PI.py

# Kamera

Hier nutzen wir ein Package von ROS direkt welches die gesammte kommunikation regelt. Hierzu gibt es hier Infos.
https://github.com/UbiquityRobotics/raspicam_node
Wir haben Camera v2 am Auto.
# PS4 Controller
Das python Script Controller_Steuerung_PS4.py ermoeglicht die Steuerung des Autos mit dem PS4 Controller. Dadurch kann sehr schnell kurz etwas getestet werden.




| :warning: TODO: überarbeiten.|
| --- |
# Old docu by nico
1. start ros master (directing and organizing all the communication) with `roscore` somewhere. Car should connect with master and master with connect. Same for camera and PC.
2. Auto hat ein Topic (streamt z.B. bild), die kann der PC abfragen (list -> [Bild Auto])

Aufm PI läuft:
RPY_Listener_motor.py (ROS_Listener_control_R_PI.py)

test_ros_launch: schickt signal an den motor
IMU_ROS_NODE: Imu auslesen
camera_listener: 




# Tmux
if you want to use tmux:

To get and build the latest from version control:

    $ git clone https://github.com/tmux/tmux.git
    $ scp -r tmux ubuntu@rospi:tmux

Now ssh into the pi with `ssh ubuntu@rospi` and execute

    $ cd tmux
    $ sh autogen.sh
    $ ./configure && make
