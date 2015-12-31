package SelectData;

import Jama.EigenvalueDecomposition;
import Jama.Matrix;

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

    private double[][] computeVariance(double[][] patterns) {
        int samples = patterns.length;
        double[][] meanAndVariance = new double[2][patterns[0].length];
        double mean;
        for(int i = 0; i < patterns[0].length; i++) {
            mean = 0.0;
            for(int j = 0; j < patterns.length; j++) {
                mean += patterns[j][i];
            }
            meanAndVariance[0][i] = mean/samples;
            for(int j = 0; j < patterns.length; j++) {
                meanAndVariance[1][i] = Math.pow(patterns[j][i] - mean, 2.0);
            }
            meanAndVariance[1][i] = meanAndVariance[1][i]/samples;
        }
        return meanAndVariance;
    }
}
