# 2D ComputerVision

Projektaufgaben des Faches 2D ComputerVision im Sommersemester 2022 bei Herr Franz.
Das Fach bestand aus Ergänzenden Übungsaufgaben zum besprochenen Vorlesungsinhalt der 2-Dimensionalen Bildverarbeitung.
Die Aufgaben wurden in Python-Notebooks mit der Ensprechenden Nummerierung bearbeitet,
die Ordner enthalten nur die Aufgabenstellungen und Testbilder die wir zur verfügung gestellt bekommen haben.

Die Präsentation zum Abschlussprojekt, dem Verkehrsschilderkenner, findet man unter folgendem Link:
[Präsentation](https://docs.google.com/presentation/d/1hsRV4n0E0FtHxk373kjO49_Q6CPppVVMucQHaqGOmZg/edit?usp=sharing 
"Verkehrsschilderkenner - 2D ComputerVision Abschlussprojekt")


## Aufgabensinhalte
util.py: Funktionensammlung aller Aufgabenteile für einen leichteren Import in den Python-Notebooks
<table>
    <thead>
        <tr>
            <th>Aufgabe</th>
            <th>Inhalt</th>
        </tr>
    </thead>
    <tbody>
        <tr>
            <td>Ex 1</td>
            <td>
                <ol type = "1">
                    <b>Farbbilder und Spiegelung</b>
                    <li> Vertraut machen mit den Python Bibliotheken cv2, scikit-image und Numpy </li>
                    <li> Zugriff auf die Farbkanäle, Daten und Informationen eines Bildes </li>
                    <li> Spiegelung eines Bildes horizontal oder vertikal </li>
                    <b>Histogramme, Binning und Lookup-Tabellen</b>
                    <li> Ein beliebiges Bild in ein 8-Bit-Graustufenbild(256 mögliche Farbwerte) umwandeln </li>
                    <li> Das Histogramm eines 8-Bit-Graustufenbildes berechnen durch Binning </li>
                    <li> Beantwortung von Fragen zur Analyse der Histogramme </li>
                    <li> Aufhellung eines Bildes mithilfe eines Lookup-Tables </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 2</td>
            <td>
                <ol type = "1">
                    <b>Histogrammanpassung</b>
                    <li> Das kumulative Histogramm eines 8-Bit-Graustufenbildes berechnen </li>
                    <li> Beantwortung von Fragen zu Punkt- und Filteroperationen </li>
                    <li> Ein Bild mittels Hisogrammanpassung an ein anderes Bild angleichen </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 3</td>
            <td>
                <ol type = "1">
                    <b>Lineare Nachbarschaftsfilter</b>
                    <li> Implementierung einer Funktion die frei wählbare Filtermasken auf ein Bild anwendet </li>
                    <li> Erweiterung der vorigen Aufgabe um eine Randbehandlung </li>
                    <li> Beantwortung von Fragen zu Filtern </li>
                    <b>Nichtlineare Nachbarschaftsfilter</b>
                    <li> Implementierung eines Medianfilters </li>
                    <li> Beantwortung von Fragen zu Medianfiltern </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 4</td>
            <td>
                <ol type = "1">
                    <b>Kantendetektion</b>
                    <li> Beantwortung von Fragen zu Kantendetektion </li>
                    <li> Implementierung des Sobel-Operators </li>
                    <li> Beantwortung von Fragen zum Sobel-Operator </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 5</td>
            <td>
                <ol type = "1">
                    <b>Hough-Transformation</b>
                    <li> Implementierung der Hough-Transformation für Geraden mit der Hesse'schen Normalform (HNF) </li>
                    <li> Erweiterung der vorigen Aufgabe um eine Schwelloperation (Threshold) </li>
                    <li> Beantwortung von Fragen zur Analyse des Ergebnisses </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 6</td>
            <td>
                <ol type = "1">
                    <b>Morphologische Filter</b>
                    <li> Implementierung der morphologischen Operationen Edode und Dilate </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Ex 7</td>
            <td>
                <ol type = "1">
                    <b>Regionenmarkierung in Binärbildern</b>
                    <li> Implementierung der sequentiellen Regionenmarkierung für Binärbilder </li>
                </ol>
            </td>
        </tr>
        <tr>
            <td>Abschlussprojekt</td>
            <td>
                <b>Verkehrsschilderkenner</b>
                Verkehrsschilderkenner.ipynb: Implementierung einer Verkehrsschild-detektion + Histogramm-Matching Klassifizierungsversuch
                elipse_detection_v2: Neuer/Anderer Versuch zur Elipsendetektion in Bildern
                template_matching_example.ipynb: Implementierung eines Template Matchings zur Klassifizierung der Ellipsen/potentiellen Schildern
                sift_example: Versuch einer Sift Implementierung zur Klassifizierung der Ellipsen/potenziellen Schilder
                util_verkehrsschilderkenner.py: Funktionensammlungen des Verkehrsschilderkenners für die Live-präsentation in complete_routine.ipynb
                complete_routine.ipynb: Sammlung aller Funktionen für die Live-Vorführung während der Präsentation
            </td>
        </tr>
    </tbody>
</table>


