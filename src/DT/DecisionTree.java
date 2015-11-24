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
     * Methode legt einen neuen DT an und trainiert ihn mit den übergebenen Daten
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param deep Maximale Tiefe des Baums
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
        defaultLabel = computeStrongestLabel(labels);
        distribution = computeClassDistribution(labels);
        numberOfLabels = labels.length;

        root = build(transpose(merge), null, featureSplit, deep, binary);
    }

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
            double maxIG = Double.NEGATIVE_INFINITY;
            double[] ig;
            int maxIGFeature = Integer.MIN_VALUE;
            int maxIGBounder = Integer.MIN_VALUE;

            if (binary == true) {
                for(int i = 0; i < data.length -1; i++) {
                    ig = computeInformationGain(data, i, binary);
                    for (int j = 0; j < ig.length; j++) {
                        System.out.print(ig[j] + " ");
                        if(ig[j] > maxIG && !usedFeature.contains(i)) {
                            maxIG = ig[j];
                            maxIGFeature = i;
                            maxIGBounder = j;
                        }
                    }
                    System.out.println();
                }
                usedFeature.add(maxIGFeature);


                deep--;
                node.setDecisionAttribute(maxIGFeature);
                node.setDecisionValueBound(values[maxIGBounder]);

                double[][][] newData = splitData(data, maxIGFeature, values[maxIGBounder]);

                System.out.println("Selected Feature: " + maxIGFeature + " IG: " + maxIG + " Deep: " + deep);
                for (int i = 0; i < 2; i++) {
                    System.out.println("Verteilung: " + newData[i][0].length);
                }

                node.left = build(newData[0], node, children, deep, binary);
                node.right = build(newData[1], node, children, deep, binary);
            }
            else  {
                for(int i = 0; i < data.length -1; i++) {
                    ig = computeInformationGain(data, i, binary);
                    if(ig[0] > maxIG && !usedFeature.contains(i)) {
                        maxIG = ig[0];
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
                    System.out.println("Verteilung: " + newData[i][0].length);
                }

                for (int i = 0; i < featureSplit; i++) {
                    node.children[i] = build(newData[i], node, children, deep, binary);
                }
            }
            return node;
        }
    }



    private int[] computeClassDistribution(double[][] data) {
        int numberOfLabels = data[0].length;
        int maxLabel = computeMaxLabel(data[data.length-1]);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)data[data.length-1][i]]++;
        }
        return distribution;
    }

    private int[] computeClassDistribution(double[] labels) {
        int numberOfLabels = labels.length;
        int maxLabel = computeMaxLabel(labels);
        int [] distribution = new int[maxLabel+1];
        for (int i = 0; i < numberOfLabels; i++) {
            distribution[(int)labels[i]]++;
        }
        return distribution;
    }


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

    private double[][] transpose(double[][] data) {
        double[][] transpose = new double[data[0].length][data.length];
        for (int i = 0; i < transpose.length; i++) {
            for (int j = 0; j < transpose[0].length; j++) {
                transpose[i][j] = data[j][i];
            }

        }
        return transpose;
    }


    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[featureSplit][];
        double upperBound, lowerBound;
        int count, offset = 0;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = values[i];
            upperBound = values[i+1];
            //upperBound = quantifyValues[featureNumber][i+1];
            //lowerBound = quantifyValues[featureNumber][i];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j+offset];
            }
            offset += count;
        }
        return subLabels;
    }

    private double[] computeInformationGain(double[][] data, int featureNumber, boolean binary) {
        double[] labels = data[data.length-1];
        double[][] subLabels;
        double probability;
        double[] gain;
        subLabels = computeSubLabels(data, featureNumber);

        if (binary == true) {
            gain = new double[featureSplit];
            for (int j = 0; j < featureSplit; j++) {
                if(subLabels[j].length != 0) {
                    gain[j] = computeEntropy(labels);
                    probability = ((double)subLabels[j].length)/((double)labels.length);
                    gain[j] -= (probability) * computeEntropy(subLabels[j]);
                }

            }

        }
        else {
            gain = new double[1];

            gain[0] = computeEntropy(labels);
            for (int j = 0; j < featureSplit; j++) {
                probability = ((double)subLabels[j].length)/((double)labels.length);
                gain[0] -= (probability) * computeEntropy(subLabels[j]);
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
     * Methode berechnet die Verteilung der Labels
     * @param labels Label/Sublabel, auf der die Verteilung berechnet wird.
     * @return int-Array mit der Labelverteilung
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
