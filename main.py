################################################################################################################################################
#                                                                                                                                              #
#   Autor: Dr. A. Schelle (alexej.schelle.ext@iu.org). Copyright : IU Internationale Hochschule GmbH, Juri-Gagarin-Ring 152, D-99084 Erfurt    #
#                                                                                                                                              #
################################################################################################################################################

# PYTHON ROUTINE zur Steuerung der GAN-Modelle und Berechnung von Immersionslevel

import os
import sys
import math
import pylab
import matplotlib.pyplot as plt
import subprocess

from GAN-DataModeler import Generator

# Ein Beispiel zur Integration von 'GAN-DataModeler.py'

with open("GAN-DataModeler.py") as file:
        
    code = file.read()
    exec(code)

# TO DOs: Implementiere die einzelnen Rechenschritte, welche in den Files 'GAN-Modeler.py', 'GAN-Estimator', und 'VR-Experience' gespeichert sind.

# Workflow:

# 1. ) Eingabe der Parameter (UserID, Age, Gender, VRHeadset, Duration)
# 2. ) Modellierung von Daten mit GAN-DataModeler.py
# 3. ) Abschätzung von Werten für MotionSickness und Immersionslevel
# 4. ) Auslagerung von MotionSickness und Immersionslevel in ein externes Files oder MySQL-Datenbank
# 5. ) Lege eine Schnittstelle (MySQL-Datenbank fest), welche für unterschiedliche Nutzer die berechneten Parameter speichert