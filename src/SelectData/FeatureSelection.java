package SelectData;

import DT.MultiSplitNode;
import Jama.EigenvalueDecomposition;
import Jama.Matrix;

import java.io.*;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 30.12.15.
 */
public class FeatureSelection {
    private Matrix pcaMatrix;
    private ArrayList<Integer> thresholdFeatures;


    public double[][] computePCA(double[][] patterns, int k) {
        double[][] covariance = computeCovarianceMatrix(patterns);
        Matrix covarianceMatrix = new Matrix(covariance);
        EigenvalueDecomposition evD = new EigenvalueDecomposition(covarianceMatrix);

        double [][] ev = evD.getV().getArray();
        double[][] pca = new double[ev.length][k];

        int evIndex = ev[0].length - 1;
        for(int i = 0; i < k; i++) {
            for(int j = 0; j < ev.length; j++) {
                pca[j][i] = ev[j][evIndex];
            }
            evIndex--;
        }
        patterns = computeZeroMeanPatterns(patterns);
        Matrix patternsMatrix = new Matrix(patterns);
        pcaMatrix = new Matrix(pca);

        return (patternsMatrix.times(pcaMatrix).getArray());
    }

    public double[][] usePCA(double[][] patterns) {
        patterns = computeZeroMeanPatterns(patterns);
        Matrix patternsMatrix = new Matrix(patterns);
        return (patternsMatrix.times(pcaMatrix).getArray());
    }

    private double[][] computeCovarianceMatrix(double[][] patterns) {
        double samples = patterns.length;
        double[][] covariance = new double[patterns[0].length][patterns[0].length];
        double median;
        for(int i = 0; i < patterns[0].length; i++) {
            median = 0;
            for(int j = 0; j < patterns.length; j++) {
                median += patterns[j][i];
            }
            median = median/samples;
            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = patterns[j][i] - median;
            }
        }
        for(int j = 0; j < patterns[0].length; j++) {
            for (int k = 0; k <= j; k++) {
                for(int l = 0; l < patterns.length; l++) {
                    covariance[j][k] += patterns[l][j]*patterns[l][k];
                }
                covariance[k][j] = covariance[j][k] = covariance[j][k]/samples;

            }
        }
        return covariance;
    }

    private double[][] computeZeroMeanPatterns(double[][] patterns) {
        int numberOfInstances = patterns.length;
        double mean;
        for(int i = 0; i < patterns[0].length; i++) {
            mean = 0.0;
            for(int j = 0; j < patterns.length; j++) {
                mean = mean + patterns[j][i];
            }
            mean = mean/(double)numberOfInstances;
            for(int j = 0; j < patterns.length; j++) {
                patterns[j][i] = patterns[j][i] - mean;
            }
        }
        return patterns;
    }

    public void saveDataPCA() {
        double[][] pca = pcaMatrix.getArray();
        try {
            FileWriter fw = new FileWriter("PCA.csv");
            for(int i = 0; i < pca.length; i++) {
                for(int j = 0; j < pca[0].length; j++) {
                    fw.append(Double.toString(pca[i][j]));
                    fw.append(",");
                }
                fw.append("\n");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public void loadDataPCA() {
        try {
            BufferedReader br = new BufferedReader(new FileReader("PCA.csv"));
            ArrayList<String> data = new ArrayList<>();
            String line;
            while ((line = br.readLine()) != null) {
                data.add(line);
            }
            String[] parts;
            double [][] pca = new double[data.size()][];
            for(int i = 0; i < pca.length; i++) {
                parts = data.get(i).split(",");
                pca[i] = new double[parts.length];
                for(int j = 0; j < pca[i].length; j++) {
                    pca[i][j] = Double.parseDouble(parts[j]);
                }
            }
            br.close();
            pcaMatrix = new Matrix(pca);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }

    public double[][] computeVarianceThreshold(double[][] patterns, double threshold) {
        thresholdFeatures = new ArrayList<>();
        double[] variance = computeVariance(patterns);
        for(int i = 0; i < variance.length; i++) {
            if(variance[i] > threshold) {
                thresholdFeatures.add(i);
            }
        }
        double[][] newPatterns = new double[patterns.length][thresholdFeatures.size()];
        int featureNumber;
        for(int i = 0; i < newPatterns[0].length; i++) {
            featureNumber = thresholdFeatures.get(i);
            for(int j = 0; j < newPatterns.length; j++) {
                newPatterns[j][i] = patterns[j][featureNumber];
            }
        }
        return newPatterns;
    }

    public double[][] useVarianceThreshold(double[][] patterns) {
        double[][] newPatterns = new double[patterns.length][thresholdFeatures.size()];
        int featureNumber;
        for(int i = 0; i < newPatterns[0].length; i++) {
            featureNumber = thresholdFeatures.get(i);
            for(int j = 0; j < newPatterns.length; j++) {
                newPatterns[j][i] = patterns[j][featureNumber];
            }
        }
        return newPatterns;
    }

    private double[] computeVariance(double[][] patterns) {
        int samples = patterns.length;
        double[] variance = new double[patterns[0].length];
        double mean;
        for(int i = 0; i < patterns[0].length; i++) {
            mean = 0.0;
            for(int j = 0; j < patterns.length; j++) {
                mean += patterns[j][i];
            }
            mean = mean/samples;
            for(int j = 0; j < patterns.length; j++) {
                variance[i] += Math.pow(patterns[j][i] - mean, 2.0);
            }
            variance[i] = variance[i]/samples;
        }
        return variance;
    }

    public void saveThresholdVariance() {
        try {
            FileWriter fw = new FileWriter("ThresholdVariance.csv");
            for(int i = 0; i < thresholdFeatures.size(); i++) {
                fw.append(Integer.toString(thresholdFeatures.get(i)));
                fw.append(",");
            }
            fw.close();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }


    public void loadThresholdVariance() {
        thresholdFeatures = new ArrayList<>();
        try {
            BufferedReader br = new BufferedReader(new FileReader("ThresholdVariance.csv"));
            String line;
            line = br.readLine();
            String[] parts = line.split(",");
            for(int i = 0; i < parts.length; i++) {
                thresholdFeatures.add(Integer.parseInt(parts[i]));
                System.out.println(thresholdFeatures.get(i));
            }
            br.close();
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }
    }
}
