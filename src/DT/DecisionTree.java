package DT;

import com.sun.tools.doclets.formats.html.SourceToHTMLConverter;

import java.util.Arrays;
import java.util.Comparator;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 13.11.15.
 */
public class DecisionTree {

    private int featureSplit;
    private Node root;
    private List<Integer> usedFeature = new LinkedList<>();
    private List<Node> leafs = new LinkedList<>();
    private double[][] quantifyValues;
    /**
     * Methode legt einen neuen DT an und trainiert ihn mit den übergebenen Daten
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param featureSplit Anzahl der Aufteilungen der einen Features
     * @param deep Maximale Tiefe des Baums
     */
    public void train (double[][] patterns, double[] labels, int featureSplit, int deep) {
        this.featureSplit = featureSplit;
        //root = build(patterns, labels, null, deep);
        quantifyData(patterns);
        double[][] merge = merge(patterns, labels);
        root = build(transpose(merge), null, featureSplit, deep);
    }

    public Node build (double[][] data, Node parent, int children, int deep) {
        Node node = new Node();
        node.parent = parent;
        node.children = new Node[children];

        if(data[0].length == 0) {
            System.out.println("Error");
            return null;
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
            System.out.println("Leafe (pure)");

            node.setLeaf(true);
            node.setClassLabel(data[data.length-1][0]);
            node.children = null;
            leafs.add(node);
            return node;
        }
        else {
            System.out.println("Node");
            node.setLeaf(false);
            double maxIG = Double.NEGATIVE_INFINITY, ig;
            double[] informationGain = new double[data.length-1];
            int maxIGFeature = Integer.MIN_VALUE;
            for(int i = 0; i < data.length -1; i++) {
                informationGain[i] = ig = computeInformationGain(data, i);
                if(ig > maxIG) {
                    maxIG = ig;
                    maxIGFeature = i;
                }
            }
            deep--;
            node.setDecisionAttribute(maxIGFeature);
            System.out.println("Selected Feature: " + maxIGFeature + " Deep: " + deep);
            double[][][] newData = splitData(data, maxIGFeature);
            for (int i = 0; i < featureSplit; i++) {
                node.children[i] = build(newData[i], node, children, deep);
            }

            return node;
        }
    }

    /**
     * Methode erstellt einen neuen Konten wenn die maximale Tiefe noch nicht erreicht wurde oder das Label noch nicht pure ist,
     * ansonsten erstellt die Methode ein Blatt
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param parent Vorgängerkonten/Elternknoten
     * @param countDeep Tiefe des Baums
     * @return neuen Knoten
     */
    public Node build (double[][] patterns, double[] labels, Node parent, int countDeep) {

        if(labels.length == 0) {
            System.out.println("Error");
            return null;
        }
        else if (countDeep == 0) {
            Node node = new Node();
            node.parent = parent;
            node.setDecisionValueUpperBound(Double.NEGATIVE_INFINITY);
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
            leafs.add(node);
            return node;
        }
        else if(isNodePure(labels) == true) {
            Node node = new Node();
            node.parent = parent;
            node.setDecisionValueUpperBound(Double.NEGATIVE_INFINITY);
            node.setDecisionAttribute(Integer.MIN_VALUE);
            node.setClassLabel(labels[0]);
            System.out.println("Leaf");
            node.setLeaf(true);
            node.left = null;
            node.right = null;
            leafs.add(node);
            return node;
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
                    if(entropyReduction[i][j] < minEntropy) {
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
            node.setDecisionValueUpperBound(value);
            List<double[][]> newData = splitData(featureIndex, value, patterns, labels);
            double [][] leftPatterns = newData.get(0);
            double [][] leftLabels = newData.get(1);
            double [][] rightPatterns = newData.get(2);
            double [][] rightLabels = newData.get(3);

            System.out.println("left: " + leftPatterns.length + " " + leftLabels[0].length);
            System.out.println("right: " + rightPatterns.length + " " + rightLabels[0].length);
            System.out.println();

            countDeep--;
            node.left = build(leftPatterns, leftLabels[0], node, countDeep);
            node.right = build(rightPatterns, rightLabels[0], node, countDeep);
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
                if (min > patterns[j][i]) {
                    min = patterns[j][i];
                }
            }
            splitSize = (max - min)/((double) featureSplit);
            for(int j = 0; j < featureSplit-1; j++) {
                featureAttribute[i][j] = min + splitSize * (double)(j +1);
            }
        }
        return featureAttribute;
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

    public double[][] merge(double[][] patterns, double[] labels) {
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


    public double[][] sort(double[][] data, int featureNumber) {
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

    public void quantifyData(double[][] patterns) {
        quantifyValues = new double[patterns[0].length][featureSplit+1];
        double max, min;
        double quantify;
        for (int i = 0; i < patterns[0].length; i++) {
            max = Double.NEGATIVE_INFINITY;
            min = Double.POSITIVE_INFINITY;
            for (int j = 0; j < patterns.length; j++) {
                if(patterns[j][i] < min) {
                    min = patterns[j][i];
                }
                if(patterns[j][i] > max) {
                    max = patterns[j][i];
                }
            }
            quantify = (max - min)/((double)featureSplit);
            for (int j = 0; j <= featureSplit; j++) {
                if(j == 0) {
                    quantifyValues[i][j] = Double.NEGATIVE_INFINITY;
                }
                else if (j == featureSplit) {
                    quantifyValues[i][j] = Double.POSITIVE_INFINITY;
                }
                else {
                    quantifyValues[i][j] = min + quantify*(double)j;
                }
            }
        }
    }

    private double[][] computeSubLabels(double[][] data, int featureNumber) {
        double[][] subLabels = new double[featureSplit][];
        double[] selectedFeature = selectFeature(data, featureNumber);
        double upperBound, lowerBound;
        int count, index;
        for (int i = 0; i < featureSplit; i++) {
            upperBound = quantifyValues[featureNumber][i];
            lowerBound = quantifyValues[featureNumber][i+1];
            count = countHitValue(selectedFeature,lowerBound, upperBound);
            subLabels[i] = new double[count];
            index = 0;
            for (int j = 0; j < selectedFeature.length; j++) {
                if (lowerBound < selectedFeature[j] && selectedFeature[j] <= upperBound) {
                    subLabels[i][index] = data[data.length-1][j];
                    index++;
                }
            }
        }
        return subLabels;
    }

    private double computeInformationGain(double[][] data, int featureNumber) {
        double[] labels = data[data.length-1];
        int numberOfLabels = labels.length;
        double[][] subLabels;
        double probability;

        double gain = computeEntropy(labels, numberOfLabels);
        subLabels = computeSubLabels(data, featureNumber);

        for (int j = 0; j < featureSplit; j++) {
            probability = ((double)subLabels[j].length)/((double)labels.length);
            gain -= (probability) * computeEntropy(subLabels[j], numberOfLabels);
        }
        return gain;
    }

    private double[][][] splitData(double[][] data, int featureNumber) {
        double[][] dataSort = sort(data, featureNumber);
        double[][][] splitData = new double[featureSplit][][];
        int[] distribution = new int[featureSplit];
        double upperBound, lowerBound;
        for (int i = 0; i < featureSplit; i++) {
            lowerBound = quantifyValues[featureNumber][i];
            upperBound = quantifyValues[featureNumber][i+1];
            distribution[i] = countHitValue(data[featureNumber], lowerBound, upperBound);
            System.out.println("distribution: " + distribution[i]);
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
            count = countHitValue(selectedFeature, Double.NEGATIVE_INFINITY, upperBound);
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
        int count = countHitValue(selectedFeature, Double.NEGATIVE_INFINITY, value);
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





    /**
     * Methode berechnet die Entropy für die einzelnen Features
     * @param patterns Train-Patterns zur berechnung der Sublabels
     * @param labels Train-Labels zur Berechnung der Entropy
     * @param featureAttribute Attribute zur Aufteilung in Sublabels
     * @return double-2d-Array mit den Entropywerten
     */
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
    private int countHitValue(double [] selectedFeature, double lowerBound, double upperBound) {
        int count = 0;
        for (int i = 0; i < selectedFeature.length; i++) {
            if (lowerBound < selectedFeature[i] && selectedFeature[i] <= upperBound) {
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
            if (pattern[node.getDecisionAttribute()] <= node.getDecisionValueUpperBound()) {
                node = node.left;
            } else {
                node = node.right;
            }
        }
        classified = node.getClassLabel();
        return classified;
    }

}
