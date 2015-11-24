package DT;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.awt.*;
import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    private boolean binary;
    private int featureSplit;
    private Node root;
    private List<Integer> usedFeature = new LinkedList<>();
    private List<Node> leafs = new LinkedList<>();
    private double[][] quantifyValues;
    private int defaultLabel;
    private int[] distribution;
    private int numberOfLabels;
    private double[] values = {Double.NEGATIVE_INFINITY, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};

    /**
     * Methode trainiert den Decision Tree
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param deep Tiefe des Baums
     * @param binary Binaerbaum
     */
    public void train (double[][] patterns, double[] labels, int deep, boolean binary) {
        this.binary = binary;
        patterns = standardization(patterns);
        featureSplit = values.length - 1;



//        int[] distribution;
//        double upperBound, lowerBound;
//        for (int h = 0; h < patterns[0].length; h++) {
//            System.out.println("Feature: " + h);
//            distribution = new int[featureSplit];
//            int sum = 0;
//            for (int i = 0; i < featureSplit; i++) {
//
//                upperBound = values[i+1];
//                lowerBound = values[i];
//                for (int j = 0; j < patterns.length; j++) {
//                    if (lowerBound <= patterns[j][h] && patterns[j][h] < upperBound) {
//                        distribution[i]++;
//                    }
//                }
//                sum += distribution[i];
//                System.out.println("Verteilung " + i + ": " + distribution[i]);
//            }
//            System.out.println("Summe " + sum);
//        }


        double[][] merge = merge(patterns, labels);

        for (int i = 0; i < transpose(merge).length-1; i++) {
            computeGiniIndex(transpose(merge), i);
        }

        defaultLabel = computeStrongestLabel(labels);
        distribution = computeClassDistribution(labels);
        numberOfLabels = labels.length;

        root = build(transpose(merge), null, featureSplit, deep, binary);
    }

    /**
     * Methode berechnet von allen Features die Standardisierung
     * @param patterns Train-Patterns
     * @return standardisierte Train-Patterns
     */
    private double[][] standardization(double[][] patterns) {

        double samples = (double)patterns.length;
        double median;
        double standardDeviation;
        for (int i = 0; i < patterns[0].length; i++) {
            median = 0;
            standardDeviation = 0;
            for (int j = 0; j < patterns.length; j++) {
                median += patterns[j][i];
            }
            median= median/samples;
            for (int j = 0; j < patterns.length; j++) {
                standardDeviation += Math.pow(patterns[j][i]-median, 2);
            }
            standardDeviation = Math.sqrt(standardDeviation/samples);

            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = (patterns[j][i] - median)/standardDeviation;
            }

        }
        return patterns;
    }

    /**
     * Methode baut entweder einen neuen Knoten oder ein Blatt
     * @param data Train-Daten (Patterns + Labels)
     * @param parent Elternknoten
     * @param children Anzahl der Kinderknoten
     * @param deep Tiefe des Baums
     * @param binary Binärbaum
     * @return Knoten oder Blatt
     */
    public Node build (double[][] data, Node parent, int children, int deep, boolean binary) {
        Node node = new Node();
        node.parent = parent;
        node.children = new Node[children];

        if(data[0].length == 0) {
            System.out.println("Leaf (patterns = 0)");
            node.setLeaf(true);
            int label = defaultLabel;
            for(int i = 0; i < distribution.length; i++) {
                if((double)distribution[i]/(double)numberOfLabels > Math.random() && i != defaultLabel) {
                    label = i;
                }
            }
            System.out.println("Label: " + label);
            node.setClassLabel((double) label);
            leafs.add(node);

            return node;
        }
        else if(deep <= 0) {
            System.out.println("Leaf (deep = 0)");
            int[] distribution = computeClassDistribution(data);
            int maxDistribution = Integer.MIN_VALUE;
            int  maxLabel = 0;
            for (int i = 0; i < distribution.length; i++) {

                if (maxDistribution < distribution[i]) {
                    maxDistribution = distribution[i];
                    maxLabel = i;
                }
            }
            node.setLeaf(true);
            node.setClassLabel(maxLabel);
            node.children = null;
            leafs.add(node);
            return node;
        }
        else if(isNodePure(data[data.length-1])) {
            System.out.println("Leaf (pure)");

            node.setLeaf(true);
            node.setClassLabel(data[data.length-1][0]);
            node.children = null;
            leafs.add(node);
            return node;
        }
        else {
            System.out.println("Node");
            node.setLeaf(false);


            if (binary == true) {
                double [] giniIndex;
                double minGI = Double.POSITIVE_INFINITY;
                int minGIFeature = Integer.MIN_VALUE;
                int minGIBound = Integer.MIN_VALUE;

                for(int i = 0; i < data.length -1; i++) {
                    giniIndex = computeGiniIndex(data, i);
                    for (int j = 0; j < giniIndex.length; j++) {

                        if(giniIndex[j] < minGI) {
                            minGI = giniIndex[j];
                            minGIFeature = i;
                            minGIBound = j;
                        }
                    }
                }
                usedFeature.add(minGIFeature);

                double value = values[minGIBound];

                deep--;
                node.setDecisionAttribute(minGIFeature);
                node.setDecisionValueBound(value);

                double[][][] newData = splitData(data, minGIFeature, value);

                System.out.println("Selected Feature: " + minGIFeature + " GI: " + minGI + " Deep: " + deep);
                for (int i = 0; i < 2; i++) {
                    System.out.println("Verteilung: " + newData[i][0].length);
                }

                node.left = build(newData[0], node, children, deep, binary);
                node.right = build(newData[1], node, children, deep, binary);
            }
            else  {
                double maxIG = Double.NEGATIVE_INFINITY;
                int maxIGFeature = Integer.MIN_VALUE;
                double informationGain;
                for(int i = 0; i < data.length -1; i++) {
                    informationGain = computeInformationGain(data, i);
                    if(informationGain > maxIG && !usedFeature.contains(i)) {
                        maxIG = informationGain;
                        maxIGFeature = i;
                    }
                }
                usedFeature.add(maxIGFeature);

                deep--;
                node.setDecisionAttribute(maxIGFeature);
                node.setDecisionValues(values);

                System.out.println("Selected Feature: " + maxIGFeature + " IG: " + maxIG + " Deep: " + deep);
                double[][][] newData = splitData(data, maxIGFeature);

                for (int i = 0; i < featureSplit; i++) {

                }

                for (int i = 0; i < featureSplit; i++) {
                    System.out.println("Verteilung: " + newData[i][0].length);
                }

                for (int i = 0; i < featureSplit; i++) {
                    node.children[i] = build(newData[i], node, children, deep, binary);
                }
            }
            return node;
        }
    }

    /**
     * Methode berechnet die Klassenverteilung
     * @param data Train-Daten (Patterns + Labels)
     * @return Verteilung der einzelnen Labels
     */
    private int[] computeClassDistribution(double[][] data) {
        int numberOfLabels = data[0].length;
        int maxLabel = computeMaxLabel(data[data.length-1]);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)data[data.length-1][i]]++;
        }
        return distribution;
    }

    /**
     * Methode berechnet die Klassenverteilung
     * @param labels Train-Labels
     * @return Verteilung der einzelnen Labels
     */
    private int[] computeClassDistribution(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }

    /**
     * Methode fuegt Patterns und Labels zusammen
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @return Train-Daten (Patterns + Labels)
     */
    private double[][] merge(double[][] patterns, double[] labels) {
        if (patterns.length != labels.length) {
            System.out.println("Patterns und Labels passen nicht zusammen!");
            return null;
        }
        double[][] merge = new double[patterns.length][patterns[0].length + 1];
        for (int i = 0; i < merge.length; i++) {
            for (int j = 0; j < merge[0].length; j++) {
                if(j == merge[0].length-1) {
                    merge[i][j] = labels[i];
                } else {
                    merge[i][j] = patterns[i][j];
                }
            }
        }
        return merge;
    }

    /**
     * Methode sortiert die Daten nach nach einem ausgewaehlten Feature aufsteigend
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber Ausgewaehltes Feature
     * @return sortierte Train-Daten (Patterns + Labels)
     */
    private double[][] sort(double[][] data, int featureNumber) {
        double[][] transpose = transpose(data);
        Arrays.sort(transpose, (double1, double2) -> {
            Double numOfKeys1 = double1[featureNumber];
            Double numOfKeys2 = double2[featureNumber];
            return numOfKeys1.compareTo(numOfKeys2);
        });
        data = transpose(transpose);
        return data;
    }

    /**
     * Methode transponiert die Daten (Matrix-Transposition)
     * @param data Train-Daten (Patterns + Labels)
     * @return transponierte Train-Daten (Patterns + Labels)
     */
    private double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }

        }
        return transpose;
    }

    /**
     * Methode berechnet die Sublabels eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Sublabels
     */
    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[featureSplit][];
        double upperBound, lowerBound;
        int count, offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j+offset];
            }
            offset += count;
        }
        return subLabels;
    }

    /**
     * Methode berechnet den Gini Index eines ausgewaehlten Features (Binaerbaum)
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Array mit Gini Indexes
     */

    private double[] computeGiniIndex(double[][] data, int featureNumber) {
        int numberOfLabels = data[0].length;
        double[][] subLabels = computeSubLabels(data, featureNumber);
        double[] giniIndex = new double[featureSplit];
        double probability;

        for (int i = 0; i < featureSplit; i++) {
            giniIndex[i] = 1.0;
            probability = ((double)subLabels[i].length)/((double)numberOfLabels);
            if(probability < 1.0) {
                giniIndex[i] -= Math.pow(probability, 2.0);
            }
        }

        return giniIndex;
    }

    /**
     * Methode berechnet den Information Gain eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Information Gain
     */
    private double computeInformationGain(double[][] data, int featureNumber) {
        double[] labels = data[data.length-1];
        double[][] subLabels;
        double probability;
        double gain;
        subLabels = computeSubLabels(data, featureNumber);
        gain = computeEntropy(labels);
        for (int j = 0; j < featureSplit; j++) {
            probability = ((double)subLabels[j].length)/((double)labels.length);

            gain -= (probability) * computeEntropy(subLabels[j]);
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
     * Methode teilt die Daten nach ein ausgewaehltes Feature auf (Binaerbaum)
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @param value ausgewaehlter Wert
     * @return Array aus Train-Daten (Patterns + Labels)
     */
    private double[][][] splitData(double[][] data, int featureNumber, double value) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[2][][];
        int distribution;
        distribution = countHitValue(data[featureNumber], Double.NEGATIVE_INFINITY, value);

        splitData[0] = new double[data.length][distribution];
        splitData[1] = new double[data.length][data[0].length-distribution];

        for (int i = 0; i < data.length; i++) {
            for(int j = 0; j < distribution; j++) {
                splitData[0][i][j] = dataSort[i][j];
            }
            for(int j = 0; j < data[0].length-distribution; j++) {
                splitData[1][i][j] = dataSort[i][j + distribution];
            }
        }
        return splitData;
    }

    /**
     * Methode teilt die Daten nach ein ausgewaehltes Feature auf
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Array aus Train-Daten (Patterns + Labels)
     */
    private double[][][] splitData(double[][] data, int featureNumber) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[featureSplit][][];
        int[] distribution = new int[featureSplit];
        double upperBound, lowerBound;


        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            distribution[i] = countHitValue(data[featureNumber], lowerBound, upperBound);
        }
        int offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            splitData[i] = new double[data.length][distribution[i]];
            for (int j = 0; j < data.length; j++) {
                for(int k = 0; k < distribution[i]; k++) {
                    splitData[i][j][k] = dataSort[j][k + offset];
                }
            }
            offset += distribution[i];
        }

        return splitData;
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
     * Methode sucht das staerkste Label heraus
     * @param labels Train-Labels
     * @return staerkste Label
     */
    private int computeStrongestLabel (double [] labels) {
        int numberOfLabels = labels.length;
        int strongestLabel = Integer.MIN_VALUE;
        int numberOfStrongestLabel = Integer.MIN_VALUE;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];

        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        for (int i = 0; i < distribution.length; i++) {
            if(distribution[i] > numberOfStrongestLabel) {
                numberOfStrongestLabel = distribution[i];
                strongestLabel = i;
            }
        }
        return strongestLabel;
    }

    /**
     * Methode zaehlt die Sampels, bis zur oberen Grenze, die den Wert erfuellen
     * @param selectedFeature ausgewaeltes Feature
     * @param upperBound obere Grenze
     * @return Anzahl der Sampels, die die Bedingung erfuellen
     */
    private int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (lowerBound <= selectedFeature[i] && selectedFeature[i] < upperBound) {
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
        patterns = standardization(patterns);
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
        double classified, upperBound, lowerBound;

        int feature;
        if (binary == true) {
            double value;
            while (node.getLeaf() == false) {
                feature = node.getDecisionAttribute();
                value = node.getDecisionValueBound();
                if (pattern[feature] < value) {
                    node = node.left;
                }
                else {
                    node = node.right;
                }
            }

        }
        else {
            double[] values;

            while (node.getLeaf() == false) {
                feature = node.getDecisionAttribute();
                values = node.getDecisionValues();
                for (int i = 0; i < featureSplit; i++) {
                    lowerBound = values[i];
                    upperBound = values[i + 1];
                    if (lowerBound <= pattern[feature] && pattern[feature] < upperBound) {
                        node = node.children[i];
                    }
                }
            }
        }
        classified = node.getClassLabel();
        return classified;
    }

}
