package DT.BSDT;

import DT.DecisionTree;
import DT.MSDT.MultiSplitNode;

import java.io.*;
import java.util.ArrayList;
import java.util.LinkedList;
import java.util.List;

/**
 * Created by MatthiasFuchs on 29.11.15.
 */
public class BinarySplitDT extends DecisionTree{

    private BinarySplitNode root;
    private List<BinarySplitNode> nodes = new LinkedList<>();
    private int count;
    private int deep;

    /**
     * Methode trainiert den Decision Tree
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param deep maximale Tiefe des Baums
     * @param featureSplit Anzahl der Aufteilungen der einzelnen Features (Quantisierung)
     */
    public void train (double[][] patterns, double[] labels, int deep, int featureSplit) {
        double[][] merge = merger(patterns, labels);;
        double[][] data = transpose(merge);
        this.deep = deep;

        numberOfInstances = labels.length;
        this.featureSplit = featureSplit;
        defaultLabel = computeStrongestLabel(labels);
        entireDistribution = computeClassDistribution(labels);
        root = build(data, null, 0);
    }

    /**
     * Methode baut entweder einen neuen Knoten oder ein Blatt
     * @param data Train-Daten (Patterns + Labels)
     * @param parent Elternknoten
     * @param deep Tiefe des Baums
     * @return Knoten oder Blatt
     */
    public BinarySplitNode build (double[][] data, BinarySplitNode parent, int deep) {
        BinarySplitNode binarySplitNode = new BinarySplitNode();
        binarySplitNode.parent = parent;
        if(data[0].length == 0) {
            System.out.println("Leaf (patterns = 0)");
            binarySplitNode.setLeaf(true);
            int label = defaultLabel;
            for(int i = 0; i < entireDistribution.length; i++) {
                if((double) entireDistribution[i]/(double) numberOfInstances > Math.random() && i != defaultLabel) {
                    label = i;
                }
            }
            System.out.println("Label: " + label);
            binarySplitNode.setClassLabel((double) label);
            nodes.add(binarySplitNode);

            return binarySplitNode;
        }
        else if(this.deep <= deep) {
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
            binarySplitNode.setLeaf(true);
            binarySplitNode.setClassLabel(maxLabel);
            binarySplitNode.left = null;
            binarySplitNode.right = null;
            nodes.add(binarySplitNode);
            return binarySplitNode;
        }
        else if(isNodePure(data[data.length-1])) {
            System.out.println("Leaf (pure)");

            binarySplitNode.setLeaf(true);
            binarySplitNode.setClassLabel(data[data.length - 1][0]);
            binarySplitNode.left = null;
            binarySplitNode.right = null;
            nodes.add(binarySplitNode);
            return binarySplitNode;
        }
        else {
            binarySplitNode.setLeaf(false);

            double[][] quantifyValues = computeQuantifyValues(data);
            double [][] entropyImpurity = new double[data.length-1][];


            for(int i = 0; i < entropyImpurity.length; i++) {
                entropyImpurity[i] = computeEntropyImpurity(data, quantifyValues, i);
            }

            double minImpurity = Double.POSITIVE_INFINITY;
            double minImpurityValue = Double.NEGATIVE_INFINITY;
            int minImpurityFeature = Integer.MIN_VALUE;
            for(int i = 0; i < entropyImpurity.length; i++) {
                if(minImpurity > entropyImpurity[i][0]) {
                    minImpurity = entropyImpurity[i][0];
                    minImpurityValue = entropyImpurity[i][1];
                    minImpurityFeature = i;
                }
            }
            usedFeature.add(minImpurityFeature);


            binarySplitNode.setDecisionValueBound(minImpurityValue);
            binarySplitNode.setDecisionAttribute(minImpurityFeature);
            deep++;
            double[][][] newData = splitData(data, minImpurityFeature, minImpurityValue);

            System.out.println("Selected Feature: " + minImpurityFeature + " EI: " + minImpurity  + " Value " + minImpurityValue + " Deep: " + deep);
            for (int i = 0; i < 2; i++) {
                System.out.println("Verteilung: " + newData[i][0].length);
            }


            binarySplitNode.left = build(newData[0], binarySplitNode, deep);
            binarySplitNode.right = build(newData[1], binarySplitNode, deep);
            nodes.add(binarySplitNode);
        }
        return binarySplitNode;
    }

    /**
     * Methode berechnet die Sublabels eines ausgewaehlten Features und Wert
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @param values ausgewaehlter Wert
     * @return Sublabels
     */
    private double[][] computeSubLabels(double[][] data, int featureNumber, double[] values) {
        double[][] sortData = sort(data, featureNumber);
        double[][] subLabels = new double[featureSplit][];
        double upperBound, lowerBound;
        int count;
        lowerBound = values[0];

        for (int i = 0; i < featureSplit; i++) {
            upperBound = values[i+1];
            count = countHitValue(data[featureNumber],lowerBound, upperBound);
            subLabels[i] = new double[count];
            for (int j = 0; j < subLabels[i].length; j++) {
                subLabels[i][j] = sortData[sortData.length-1][j];

            }
        }
        return subLabels;
    }

    /**
     * Methode berechnet die minimale Entropie-Unreinheit eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param quantifyValues quantifizierte Grenzwerte
     * @param featureNumber ausgewaehltes Feature
     * @return Entropie-Unreinheit und den dazugehörtigen Wert
     */
    private double[] computeEntropyImpurity(double[][] data, double[][] quantifyValues, int featureNumber) {
        double[] impurityAndValue = new double[2];
        double[][] subLabels = computeSubLabels(data, featureNumber, quantifyValues[featureNumber]);
        double[] impurity = new double[featureSplit];

        for (int i = 0; i < featureSplit; i++) {
            impurity[i] = computeEntropy(subLabels[i]);
        }
        double minImpurity = Double.POSITIVE_INFINITY;
        double value = quantifyValues[featureNumber][0];
        for (int i = 0; i < featureSplit; i++) {
            if(minImpurity > impurity[i]) {
                minImpurity = impurity[i];
                value = quantifyValues[featureNumber][i+1];
            }
        }
        impurityAndValue[0] = minImpurity;
        impurityAndValue[1] = value;
        return impurityAndValue;
    }

    /**
     * Methode teilt die Daten nach ein ausgewaehltes Feature auf
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
     * Methode geht durch den Baum durch und liefert die jeweilige Klasse zurueck wenn es auf ein Blatt trifft
     * @param pattern klassifizierendes Pattern
     * @return klassifiziertes Label
     */
    public double passTree(double[] pattern) {
        BinarySplitNode binarySplitNode = root;
        double classified;

        int feature;
        double value;
        while (binarySplitNode.getLeaf() == false) {
            feature = binarySplitNode.getDecisionAttribute();
            value = binarySplitNode.getDecisionValueBound();
            if (pattern[feature] < value) {
                binarySplitNode = binarySplitNode.left;
            }
            else {
                binarySplitNode = binarySplitNode.right;
            }
        }
        classified = binarySplitNode.getClassLabel();
        return classified;
    }

    public void saveData() {
        try {
            FileWriter fw = new FileWriter("binarySplitDT.csv");
            save(root, fw);

            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void save(BinarySplitNode node, FileWriter fw) {
        try {
            if(node.getLeaf() == false) {
                fw.append(Boolean.toString(false));
                fw.append(",");
                fw.append(Integer.toString(node.getDecisionAttribute()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append(",");
                fw.append(Double.toString(node.getDecisionValueBound()));
                fw.append(",");
                fw.append("\n");
                save(node.left, fw);
                save(node.right, fw);
            }
            else {
                fw.append(Boolean.toString(true));
                fw.append(",");
                fw.append(Double.toString(node.getClassLabel()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append("\n");
            }
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadData() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("binarySplitDT.csv"));
            ArrayList<String> data = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                data.add(line);
            }
            count = -1;
            this.root = buildWithLoadedData(null, data);
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private BinarySplitNode buildWithLoadedData(BinarySplitNode parent, ArrayList<String> data) {

        count++;
        BinarySplitNode node = new BinarySplitNode();
        node.parent = parent;
        String[] parts = data.get(count).split(",");
        if(parts[0].equals("false")) {
            node.setLeaf(false);
            node.setDecisionAttribute(Integer.parseInt(parts[1]));
            node.setDecisionValueBound(Double.parseDouble(parts[3]));
            node.left = buildWithLoadedData(node, data);
            node.right = buildWithLoadedData(node, data);
        } else {
            node.setLeaf(true);
            node.setClassLabel(Double.parseDouble(parts[1]));
            node.left = null;
            node.right = null;
        }
        return node;
    }

}
