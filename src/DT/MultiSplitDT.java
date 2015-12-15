package DT;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 29.11.15.
 */
public class MultiSplitDT extends DecisionTree {

    private MultiSplitNode root;

    private double[] values = {Double.NEGATIVE_INFINITY, -3.0, -2.0, -1.0, 0.0, 1.0, 2.0, 3.0, Double.POSITIVE_INFINITY};
    private int count;


    public MultiSplitDT() {
        featureSplit = values.length - 1;
    }

    /**
     * Methode trainiert den Decision Tree
     * @param patterns Train-Patterns
     * @param labels Train-Labels
     * @param deep maximale Tiefe des Baums
     */
    public void train (double[][] patterns, double[] labels, int deep) {
        this.deep = deep;
        patterns = standardization(patterns);
        double[][] merge;
        double[][] data;

        merge = merger(patterns, labels);
        data = transpose(merge);

        numberOfInstances = labels.length;
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
    public MultiSplitNode build (double[][] data, MultiSplitNode parent, int deep) {
        MultiSplitNode node = new MultiSplitNode();
        node.parent = parent;
        if(data[0].length == 0) {
            node.setLeaf(true);
            node.deep = deep;
            int label = defaultLabel;
            for(int i = 0; i < entireDistribution.length; i++) {
                if((double) entireDistribution[i]/(double) numberOfInstances > Math.random() && i != defaultLabel) {
                    label = i;
                }
            }
            node.setClassLabel((double) label);
            return node;
        }
        else if(this.deep <= deep) {
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
            node.deep = deep;
            node.setClassLabel(maxLabel);
            node.children = null;
            return node;
        }
        else if(isNodePure(data[data.length-1])) {
            node.setLeaf(true);
            node.deep = deep;
            node.setClassLabel(data[data.length - 1][0]);
            node.children = null;
            return node;
        }
        else {
            node.setLeaf(false);
            node.deep = deep;
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


            node.setDecisionAttribute(maxIGFeature);
            node.setDecisionValues(values);

            double[][][] newData = splitData(data, maxIGFeature);

            node.children = new MultiSplitNode[featureSplit];
            deep++;
            for (int i = 0; i < featureSplit; i++) {
                node.children[i] = build(newData[i], node, deep);
            }
            return node;
        }
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
     * Methode berechnet den Information-Gain eines ausgewaehlten Features
     * @param data Train-Daten (Patterns + Labels)
     * @param featureNumber ausgewaehltes Feature
     * @return Information-Gain
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
     * Methode geht durch den Baum durch und liefert die jeweilige Klasse zurueck wenn es auf ein Blatt trifft
     * @param pattern klassifizierendes Pattern
     * @return klassifiziertes Label
     */
    private double passTree(double[] pattern) {
        MultiSplitNode node = root;
        double classified, upperBound, lowerBound;

        int feature;
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
        classified = node.getClassLabel();
        return classified;
    }

    public void saveData() {
        try {
            FileWriter fw = new FileWriter("multiSplitDT.csv");
            save(root, fw);

            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    private void save(MultiSplitNode node, FileWriter fw) {
        try {
            if(node.getLeaf() == false) {
                fw.append(Boolean.toString(false));
                fw.append(",");
                fw.append(Integer.toString(node.getDecisionAttribute()));
                fw.append(",");
                fw.append(Integer.toString(node.deep));
                fw.append(",");
                for(int i = 0; i < node.getDecisionValues().length; i++) {
                    fw.append(Double.toString(node.getDecisionValues()[i]));
                    fw.append(",");
                }
                fw.append("\n");
                for(int i = 0; i < featureSplit; i++) {
                    save(node.children[i], fw);
                }
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
            BufferedReader br = new BufferedReader(new FileReader("multiSplitDT.csv"));
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

    private MultiSplitNode buildWithLoadedData(MultiSplitNode parent, ArrayList<String> data) {

        count++;
        MultiSplitNode node = new MultiSplitNode();
        node.parent = parent;
        String[] parts = data.get(count).split(",");
        if(parts[0].equals("false")) {
            node.setLeaf(false);
            node.setDecisionAttribute(Integer.parseInt(parts[1]));
            double[] decisionValues = new double[featureSplit +1];
            for(int i = 0; i < decisionValues.length; i++) {
                decisionValues[i] = Double.parseDouble(parts[i + 3]);
            }
            node.setDecisionValues(decisionValues);
            node.children = new MultiSplitNode[featureSplit];
            for(int i = 0; i < node.children.length; i++) {
                node.children[i] = buildWithLoadedData(node, data);
            }
        } else {
            node.setLeaf(true);
            node.setClassLabel(Double.parseDouble(parts[1]));
            node.children = null;
        }
        return node;
    }
}
