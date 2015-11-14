package DT;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    private double[][] featureAttribute;
    private int featureSplit;
    private int deep;
    private Node root;

    /**
     * Methode legt einen neuen DT an und trainiert ihn mit den übergebenen Daten
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param featureSplit Anzahl der Aufteilungen der einen Features
     * @param deep Maximale Tiefe des Baums
     */
    public void train (double[][] patterns, double[] labels, int featureSplit, int deep) {
        this.deep = deep;
        this.featureSplit = featureSplit;
        computeFeatureAttribute(patterns);
        root = build(patterns, labels, null);

    }

    /**
     * Methode erstellt einen neuen Konten wenn die maximale Tiefe noch nicht erreicht wurde oder das Label noch nicht pure ist,
     * ansonsten erstellt die Methode ein Blatt
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param parent Vorgängerkonten/Elternknoten
     * @return neuen Knoten
     */
    public Node build (double[][] patterns, double[] labels, Node parent) {
        Node node = new Node();
        node.parent = parent;
        if(isNodePure(labels) == true) {
            node.setLeaf(true);
            node.setClassLabel(labels[0]);
            return null;
        } else {
            double[][] subLabels;
            double[][] informationGain = new double[patterns[0].length][this.featureSplit];

            for (int i = 0; i < patterns[0].length; i++) {
                subLabels = computeSubLabels(patterns, labels, i);
                informationGain[i] = computeInformationGain(labels, subLabels);
            }
            int indexFeatureMaxGain = Integer.MIN_VALUE, indexFeatureValueMaxGain = Integer.MIN_VALUE;
            double maxGain = Double.NEGATIVE_INFINITY;
            for (int i = 0; i < informationGain.length; i++) {
                for (int j = 0; j < informationGain[0].length; j++) {
                    if (maxGain < informationGain[i][j]) {
                        maxGain = informationGain[i][j];
                        indexFeatureMaxGain = i;
                        indexFeatureValueMaxGain = j;
                    }
                }
            }

            node.setDecisionAttribute(indexFeatureMaxGain);
            node.setDecisionValue(featureAttribute[indexFeatureMaxGain][indexFeatureValueMaxGain]);
            List<double[][]> newData = splitData(indexFeatureMaxGain, indexFeatureValueMaxGain, patterns, labels);
            double [][] leftPatterns = newData.get(0);
            double [][] leftLabels = newData.get(1);
            double [][] rightPatterns = newData.get(2);
            double [][] rightLabels = newData.get(3);

            node.left = build(leftPatterns, leftLabels[0], node);
            node.right = build(rightPatterns, rightLabels[0], node);
            return node;
        }
    }


    /**
     * Methode teilt die einzelnen Features in Bereiche ein und speichert sie in featureAttribute ab.
     * @param patterns Train-Patterns
     */
    private void computeFeatureAttribute(double[][] patterns) {
        featureAttribute = new double[patterns[0].length][featureSplit];
        double max = Double.NEGATIVE_INFINITY;
        double min = Double.POSITIVE_INFINITY;
        double splitSize;
        for (int i = 0; i < patterns[0].length; i++) {
            for (int j = 0; j < patterns.length; j++) {
                if (max < patterns[j][i]) {
                    max = patterns[j][i];
                }
                else if (min > patterns[j][i]) {
                    min = patterns[j][i];
                }
            }
            splitSize = (max - min)/((double) featureSplit);
            for(int j = 0; j < featureSplit; j++) {
                featureAttribute[i][j] = min + splitSize * (j+1);
            }
        }
    }

    /**
     * Methode trennt die Labels in SubLabels auf zum Berechnen des Information Gains
     * @param patterns Train-Patterns zur Entscheidung der SubLevel Einteilung
     * @param labels Train-Patterns zur Einteilung in SubLabels
     * @param featureNumber Ausgewaehltes Feature nachdem SubLabels getrennt wird.
     * @return Sublabels
     */
    private double[][] computeSubLabels(double[][] patterns, double[] labels, int featureNumber) {
        double[][] subLabels = new double[featureSplit][];
        double upperBound;
        double[] selectedFeature = selectFeature(patterns, featureNumber);
        int count, index;
        for (int i = 0; i < featureSplit; i++) {
            upperBound = featureAttribute[featureNumber][i];
            count = countHitValue(selectedFeature, upperBound);
            subLabels[i] = new double[count];
            index = 0;
            for (int j = 0; j < selectedFeature.length; j++) {
                if (selectedFeature[j] < upperBound) {
                    subLabels[i][index] = labels [j];
                    index++;
                }
            }
        }
        return subLabels;
    }

    /**
     * Methode teilt die Daten in zwei Bereiche auf (rechts/links vom Knoten)
     * @param featureNumber Ausgewaeltes Feature, das die Aufteilung bestimmt
     * @param value Featurewert, das die Aufteilungsgrenze definiert
     * @param patterns Aufzuteilendes Patterns
     * @param labels Aufzuteilendes Labels
     * @return Liste mit rechten und linken Patterns/Labels
     */
    private List<double[][]> splitData(int featureNumber, double value, double[][] patterns, double[] labels) {
        List<double[][]> splitData = new LinkedList<>();
        double [] selectedFeature = selectFeature(patterns, featureNumber);
        int count = countHitValue(selectedFeature, value);
        double[][] leftPatterns = new double[count][];
        double [][] leftLabels = new double[1][count];
        double[][] rightPatterns = new double[selectedFeature.length-count][];
        double [][] rightLabels = new double[1][selectedFeature.length-count];

        int countRight = 0, countLeft = 0;

        for (int i = 0; i < selectedFeature.length; i++) {
            if(selectedFeature[i] < value) {
                leftPatterns[countLeft] = patterns[i];
                leftLabels[0][countLeft] = labels[i];
                countLeft++;
            } else {
                rightPatterns[countRight] = patterns[i];
                rightLabels[0][countRight] = labels[i];
                countRight++;
            }
        }
        splitData.add(leftPatterns);
        splitData.add(leftLabels);
        splitData.add(rightPatterns);
        splitData.add(rightLabels);
        return splitData;
    }

    /**
     * Methode selektiert ein ausgewaehltes Feature aus den Train-Patterns
     * @param patterns Train-Patterns
     * @param featureNumber Ausgewähltes Feature
     * @return Selektiertes Feature aus den Train-Patterns
     */
    private double [] selectFeature(double[][] patterns, int featureNumber) {
        double[] selectedFeature = new double[patterns.length];
        for(int i = 0; i < patterns.length; i++) {
            selectedFeature[i] = patterns[i][featureNumber];
        }
        return selectedFeature;
    }

    /**
     * Methode zur Berechnung des Informationsgains
     * @param labels Labels für die allgemeine Entropy
     * @param subLabels Sublabels auf denen der Informationsgain-Zunahme/Abnahme berechnet wird.
     * @return double-Array mit den Informationsgain zurueckgeliefert wird
     */
    private double[] computeInformationGain(double[] labels, double[][] subLabels) {
        double [] gain = new double [featureSplit];
        for (int i = 0; i < featureSplit; i++) {
            gain[i] = computeEntropy(labels);
            for (int j = 0; j < featureSplit; j++) {
                if (subLabels[j].length > 0) {
                    if (i == j) {
                        gain[i] -= (subLabels[j].length / labels.length) * computeEntropy(subLabels[j]);
                    } else {
                        gain[i] += (subLabels[j].length / labels.length) * computeEntropy(subLabels[j]);
                    }
                }
            }
        }
        return gain;
    }

    /**
     * Methode zur Berechnung der Entropy
     * @param labels Labels/Sublabels auf den die Entropy berechnet werden soll
     * @return gibt den Entropywert zurueck
     */
    private double computeEntropy(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int[] distribution = computeClassDistribution(labels);

        double entropy = Double.NEGATIVE_INFINITY;
        for (int i = 0; i <= maxLabel; i++) {
            entropy += -(distribution[i]/numberOfLabels)*(Math.log(distribution[i]/numberOfLabels)/Math.log(2.0));
        }
        return entropy;
    }

    /**
     * Methode sucht das maximale Label, dass in Labels/Sublabels vorkomm
     * @param labels Labels/Sublabels auf den das maximale Label gesucht wird
     * @return maximale Label in Labels/Sublabels
     */
    private int computeMaxLabel(double[] labels) {
        int maxLabel = 0;
        int numberOfLabels = labels.length;
        for (int i = 0; i < numberOfLabels; i++) {
            if (maxLabel < (int) labels[i]) {
                maxLabel = (int) (labels[i]);
            }
        }
        return maxLabel;
    }

    /**
     * Methode berechnet die Verteilung der Labels
     * @param labels Label/Sublabel, auf der die Verteilung berechnet wird.
     * @return int-Array mit der Labelverteilung
     */
    private int[] computeClassDistribution(double [] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        for (int i = 0; i < numberOfLabels; i++){
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }

    /**
     * Methode zaehlt die Sampels, bis zur oberen Grenze, die den Wert erfuellen
     * @param selectedFeature ausgewaeltes Feature
     * @param upperBound obere Grenze
     * @return Anzahl der Sampels, die die Bedingung erfuellen
     */
    private int countHitValue(double [] selectedFeature, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (selectedFeature[i] < upperBound) {
                count++;
            }
        }
        return count;
    }

    /**
     * Methode ueberprueft, ob ein Konten/Subset pure ist
     * @param labels Train-Label-Subset eines Knotens
     * @return true, wenn der Knoten/Subset pure ist, andernfalls false
     */
    private boolean isNodePure(double[] labels) {
        for (int i = 0; i < labels.length; i++) {
            if (labels[0] !=labels[i]) {
                return false;
            }
        }
        return true;
    }

    /**
     * Getter-Methode für Root-Knoten
     * @return Root Knoten
     */
    public Node getRoot() {
        return root;
    }

    public double[] classify(double[][] patterns) {
        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }

    public double passTree(double[] pattern) {
        Node node = root;
        double classified = Double.NEGATIVE_INFINITY;
        while (node.getLeaf() == false) {
            if (pattern[node.getDecisionAttribute()] < node.getDecisionValue()) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        classified = node.getClassLabel();
        return classified;
    }

}
