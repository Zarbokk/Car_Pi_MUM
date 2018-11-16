# Python/ROS Ansteuerung

Die Ansteuerung des Modell Autos kann auch per Python/ROS geschehen.
Folgende Datanaustausch wird hier Beschrieben:

1. Motor Steuerung
2. IMU auslesen
3. Kamera auslesen
4. Steuerung mit PS4 Controller

Dazu wird Folgendes benoetigt :
1. Linux/Mac
2. Python
3. ROS Kinetic
4. Eventuell andere Packages wie z.B. OpenCV

Auf Windows ist ROS NICHT moeglich.

Alle Python Dateien sind weder Perfekt noch zwingend richtig, und Work In Progress. Daher nicht einfach komplett Blind nutzen

ROS ist das Roboter Operating System. Hierzu gibt es sehr viele Videos Online die ROS Erklaeren.
 http://www.ros.org/

Das Image was auf die SD Karte gezogen werden muss ist hier downzuloaden: https://downloads.ubiquityrobotics.com/
Hier steht auch wie man sich mit dem PI verbindet. Ihr solltet bei erster Anwendung dafuer Sorgen, dass sich der PI Automatisch einloggt, damit automatisch das W-Lan entsteht. (Speichert das alte Image falls ihr zu Matlab zurueck wollt)


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




