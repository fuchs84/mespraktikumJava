package DT;

import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    private int featureSplit;
    private Node root;
    private List<Integer> usedFeature = new LinkedList<>();
    /**
     * Methode legt einen neuen DT an und trainiert ihn mit den übergebenen Daten
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param featureSplit Anzahl der Aufteilungen der einen Features
     * @param deep Maximale Tiefe des Baums
     */
    public void train (double[][] patterns, double[] labels, int featureSplit, int deep) {
        this.featureSplit = featureSplit;
        root = build(patterns, labels, null, deep);
    }

    /**
     * Methode erstellt einen neuen Konten wenn die maximale Tiefe noch nicht erreicht wurde oder das Label noch nicht pure ist,
     * ansonsten erstellt die Methode ein Blatt
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param parent Vorgängerkonten/Elternknoten
     * @return neuen Knoten
     */
    public Node build (double[][] patterns, double[] labels, Node parent, int count) {

        if(labels.length == 0) {
            System.out.println("Error");
            return null;
        }
        else if (count == 0) {
            Node node = new Node();
            node.parent = parent;
            node.setDecisionValue(Double.NEGATIVE_INFINITY);
            node.setDecisionAttribute(Integer.MIN_VALUE);
            int[] distribution = computeClassDistribution(labels);
            int maxDistribution = Integer.MIN_VALUE;
            int  maxLabel = 0;
            for (int i = 0; i < distribution.length; i++) {
                if (maxDistribution < distribution[i]) {
                    maxDistribution = distribution[i];
                    maxLabel = i;
                }
            }
            node.setClassLabel(maxLabel);
            System.out.println("Leaf");
            node.setLeaf(true);
            node.left = null;
            node.right = null;
            return null;
        }
        else if(isNodePure(labels) == true) {
            Node node = new Node();
            node.parent = parent;
            node.setDecisionValue(Double.NEGATIVE_INFINITY);
            node.setDecisionAttribute(Integer.MIN_VALUE);
            node.setLeaf(true);
            node.setClassLabel(labels[0]);
            System.out.println("Leaf");
            node.left = null;
            node.right = null;
            return null;
        } else {
            System.out.println("Laenge: " + labels.length);
            Node node = new Node();
            node.parent = parent;
            node.setLeaf(false);
            node.setClassLabel(Double.NEGATIVE_INFINITY);
            double[][] featureAttribute = computeFeatureAttribute(patterns);
            double [][] entropyReduction = computeEntropyReduction(patterns, labels, featureAttribute);

            double minEntropy = Double.POSITIVE_INFINITY;
            int featureIndex = Integer.MIN_VALUE, valueIndex = Integer.MIN_VALUE;

            for (int i = 0; i < entropyReduction.length; i++) {
                for (int j = 0; j < entropyReduction[0].length; j++) {
                    if(entropyReduction[i][j] < minEntropy && usedFeature.contains(i) == false
                            && !Double.isNaN(featureAttribute[i][j])) {
                        minEntropy = entropyReduction[i][j];
                        featureIndex = i;
                        valueIndex = j;
                    }
                }
            }

            usedFeature.add(featureIndex);

            System.out.println("Used Features: ");
            for(int i = 0; i < usedFeature.size(); i++) {
                System.out.print(usedFeature.get(i) + " ");
            }
            System.out.println();


            double value = featureAttribute[featureIndex][valueIndex];

            System.out.println("Feature: " + featureIndex + " Value: " + valueIndex + " Value: " + value);

            node.setDecisionAttribute(featureIndex);
            node.setDecisionValue(value);
            List<double[][]> newData = splitData(featureIndex, value, patterns, labels);
            double [][] leftPatterns = newData.get(0);
            double [][] leftLabels = newData.get(1);
            double [][] rightPatterns = newData.get(2);
            double [][] rightLabels = newData.get(3);

            System.out.println("left: " + leftPatterns.length + " " + leftLabels[0].length);
            System.out.println("right: " + rightPatterns.length + " " + rightLabels[0].length);
            System.out.println();

            count--;
            node.left = build(leftPatterns, leftLabels[0], node, count);
            node.right = build(rightPatterns, rightLabels[0], node, count);
            return node;
        }
    }


    /**
     *  * Methode teilt die einzelnen Features in Bereiche ein und speichert sie in featureAttribute ab.
     * @param patterns Train-Patterns
     * @return FeatureAttribute enthaelt die Grenzen der jeweiligen Features
     */
    private double[][] computeFeatureAttribute(double[][] patterns) {
        double [][] featureAttribute = new double[patterns[0].length][featureSplit-1];
        double max;
        double min;
        double splitSize;
        for (int i = 0; i < patterns[0].length; i++) {
            max = Double.NEGATIVE_INFINITY;
            min = Double.POSITIVE_INFINITY;
            for (int j = 0; j < patterns.length; j++) {
                if (max < patterns[j][i]) {
                    max = patterns[j][i];
                }
                else if (min > patterns[j][i]) {
                    min = patterns[j][i];
                }
            }
            splitSize = (max - min)/((double) featureSplit);
            for(int j = 0; j < featureSplit-1; j++) {
                featureAttribute[i][j] = min + splitSize * (j +1);
            }
        }
        return featureAttribute;
    }

    /**
     * Methode trennt die Labels in SubLabels auf zum Berechnen des Information Gains
     * @param patterns Train-Patterns zur Entscheidung der SubLevel Einteilung
     * @param labels Train-Patterns zur Einteilung in SubLabels
     * @param featureNumber Ausgewaehltes Feature nachdem SubLabels getrennt wird.
     * @param featureAttribute Zum Bestimmen der Grenzen
     * @return Sublabels;
     */
    private double[][] computeSubLabels(double[][] patterns, double[] labels, int featureNumber, double[][] featureAttribute) {
        double[][] subLabels = new double[featureSplit-1][];
        double upperBound;
        double[] selectedFeature = selectFeature(patterns, featureNumber);
        int count, index;
        for (int i = 0; i < featureSplit-1; i++) {
            upperBound = featureAttribute[featureNumber][i];
            count = countHitValue(selectedFeature, upperBound);
            subLabels[i] = new double[count];
            index = 0;
            for (int j = 0; j < selectedFeature.length; j++) {
                if (selectedFeature[j] <= upperBound) {
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
            if(selectedFeature[i] <= value) {
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


//    /**
//     * Methode zur Berechnung des Informationsgains
//     * @param labels Labels für die allgemeine Entropy
//     * @param subLabels Sublabels auf denen der Informationsgain-Zunahme/Abnahme berechnet wird.
//     * @return double-Array mit den Informationsgain zurueckgeliefert wird
//     */
//    private double[] computeInformationGain(double[] labels, double[][] subLabels) {
//        double [] gain = new double [featureSplit];
//        double probability = 0.0;
//        for (int i = 0; i < featureSplit; i++) {
//            gain[i] = computeEntropy(labels);
//            for (int j = 0; j < featureSplit; j++) {
//                probability = ((double)subLabels[j].length)/((double)labels.length);
//                System.out.println("Probability: " + probability);
//                if (i == j) {
//                    gain[i] -= (probability) * computeEntropy(subLabels[j]);
//                } else {
//                    gain[i] += (probability) * computeEntropy(subLabels[j]);
//                }
//            }
//        }
//        return gain;
//    }

    private double[][] computeEntropyReduction(double[][] patterns, double[] labels, double[][] featureAttribute) {
        int numberOfFeatures = patterns[0].length;
        int numberOfLabels = labels.length;
        double[][] subLabels;
        double[][] entropyReduction = new double[numberOfFeatures][featureSplit-1];
        for(int i = 0; i < numberOfFeatures; i++) {
            subLabels = computeSubLabels(patterns, labels, i, featureAttribute);
            for (int j = 0; j < featureSplit-1; j++) {
                entropyReduction[i][j] = computeEntropy(subLabels[j], numberOfLabels);
            }
        }
        return entropyReduction;
    }


    /**
     * Methode zur Berechnung der Entropy
     * @param labels Labels/Sublabels auf den die Entropy berechnet werden soll
     * @return gibt den Entropywert zurueck
     */
    private double computeEntropy(double[] labels, int numberOfLabels) {
        int maxLabel = computeMaxLabel(labels);
        int[] distribution = computeClassDistribution(labels);


        double entropy = 0.0;
        double probability;
        for (int i = 0; i <= maxLabel; i++) {
            if (distribution[i] != 0) {
                probability = ((double)distribution[i])/((double)numberOfLabels);
                entropy += -(probability) * (Math.log(probability) / Math.log(2.0));
            }
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

        for (int i = 0; i < numberOfLabels; i++) {
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
            if (selectedFeature[i] <= upperBound) {
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

    /**
     * Methode klassifiziert die uebergebenen Patterns
     * @param patterns Patterns die klassifiziert Werden
     * @return double-Array mit den jeweiligen Labels
     */
    public double[] classify(double[][] patterns) {
        double[] labels = new double[patterns.length];
        for (int i = 0; i < patterns.length; i++) {
            labels[i] = passTree(patterns[i]);
        }
        return labels;
    }

    /**
     * Methode geht durch den Baum durch und liefert die jeweilige Klasse zurück wenn es auf ein Blatt trifft
     * @param pattern zu klassifizierendes Pattern
     * @return klassifiziertes Label
     */
    public double passTree(double[] pattern) {
        Node node = root;
        double classified;
        while (node.getLeaf() == false) {
            if (pattern[node.getDecisionAttribute()] <= node.getDecisionValue()) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        classified = node.getClassLabel();
        return classified;
    }

}