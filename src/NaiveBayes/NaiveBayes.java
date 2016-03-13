package NaiveBayes;

import java.util.*;

/**
 * Naive Bayes Classifier (own implementation)
 */
public class NaiveBayes {

    //Train-features
    public double[][] trainMatrix;

    //Train-labels
    public double[] trainlabel;

    /**
     * [i][0][0] Label
     * [i][1][0] Label probability
     * [i][2+*][0] Variance
     * [8][2+*][1] Mean
     */
    public double[][][] featureprobability;


    /**
     * Method trains the Classifier employing the variance and means of the features within the classes.
     * @param patterns Feature-set
     * @param labels Label-set
     */
    public void train(double[][] patterns, double[] labels) {

        trainMatrix = new double[patterns.length][patterns[0].length];
        trainlabel = new double[labels.length];


        for (int i = 0; i < patterns.length; i++) {
            trainlabel[i] = labels[i];
            trainMatrix[i] = patterns[i];
        }

        Integer[] numbers = new Integer[trainlabel.length];
        for (int i = 0; i < trainlabel.length; i++) {
            numbers[i] = (int) trainlabel[i];
        }
        Set<Integer> uniqKeys = new LinkedHashSet<Integer>(Arrays.asList(numbers ));

        //Array with occur labels
        Integer[] labelunique = uniqKeys.toArray(new Integer[uniqKeys.size()]);

        featureprobability = new double[uniqKeys.size()][trainMatrix[0].length+2][2];
        //Split data into the individual labels
        for (int i = 0; i <labelunique.length ; i++) {
            int k=0;
            for (int j = 0; j < trainlabel.length; j++) {
                if(labelunique[i]==trainlabel[j]) k+=1;
            }
            double[][] singlelabelpattern = new double[k][trainMatrix[0].length];
            double[] singlelabellabel = new double[k];
            int t = 0;
            for (int j = 0; j <trainlabel.length ; j++) {
                if(labelunique[i]==trainlabel[j]){
                    singlelabellabel[t]=trainlabel[j];
                    for (int l = 0; l < trainMatrix[0].length; l++) {
                        singlelabelpattern[t][l]=trainMatrix[j][l];
                    }
                    t+=1;
                }

            }

            //Calculate mean and variance
            double[] mean = getMean(singlelabelpattern);
            double pmf = (double)k/trainlabel.length;
            double[] variance = getVariance(singlelabelpattern,mean);
            featureprobability[i][0][0] = labelunique[i];
            featureprobability[i][1][0] = pmf;
            for (int j = 2; j < featureprobability[0].length ; j++){

                featureprobability[i][j][0] = variance[j-2];
                featureprobability[i][j][1] = mean[j-2];

                }
            }
        }


    /**
     * Method classifies a feature-set
     * @param patterns Feature-set
     * @return Classified label-set
     */
    public double[] classify(double[][] patterns ){
        double[] predictedlabel = new double[patterns.length];
        for (int i = 0; i < predictedlabel.length; i++) {
            predictedlabel[i] = classifyGaussian(patterns[i]);

        }
        return predictedlabel;
    }

    /**
     * Method classifies the individual features (individual timestamps)
     * @param pattern Feature-vector
     * @return Classified label
     */
    private double classifyGaussian(double[] pattern){
        double[] classprobability = new double[featureprobability.length];
        for (int i = 0; i <featureprobability.length ; i++) {
            double[] featureinclass = new double[pattern.length];

            for (int j = 0; j <pattern.length ; j++) {


                //calculates the gaussian probability
                double proba =((1/Math.sqrt(2*Math.PI*Math.pow(featureprobability[i][j+2][0],2)))*
                                Math.pow(Math.E,
                                    -(Math.pow(pattern[j]-featureprobability[i][j+2][1],2)/
                                    (2*Math.pow(featureprobability[i][j+2][0],2)))));

                if (proba!=0.0&&proba>0) {
                    featureinclass[j] = Math.abs(Math.log10(proba));
                }
            }
            double entireprob = featureprobability[i][1][0];
            for (int j = 0; j < pattern.length; j++) {
                if (featureinclass[j]!=0.0&&featureinclass[j]>0) {
                    entireprob *= featureinclass[j];
               }
            }
            classprobability[i]=entireprob;
        }
        double extremum = 0;
        for (int t = 0; t < 1; t++) {
            for (int z = 0; z < classprobability.length; z++) {

                for (int w = z; w < classprobability.length; w++) {
                    if (classprobability[z] < classprobability[w]) {
                        z = w - 1;
                        break;
                    } else if (w == classprobability.length - 1) {
                        extremum = z;
                        classprobability[z] = Integer.MAX_VALUE;
                        z = classprobability.length + 1;
                        break;
                    }
                }
            }
        }
        return featureprobability[(int)extremum][0][0];
    }

    /**
     * Method calculates the mean of the features within the individual labels
     * @param patterns Feature-set
     * @return Means of the features
     */
    private double[] getMean(double[][] patterns){
        double[] mean = new double[patterns[0].length];
        for (int i = 0; i < mean.length ; i++) {
            double sum =0;
            for (int j = 0; j < patterns.length ; j++) {
                sum += patterns[j][i];
            }
            mean[i]= sum/patterns.length;
        }
        return mean;
    }


    /**
     * Method calculates the variance of the features within the individual labels
     * @param patterns Feature-set
     * @param mean Means of the features
     * @return Variance of the features
     */
    private double[] getVariance(double[][] patterns,double[]mean){
        double[] variance = new double[patterns[0].length];
        for (int i = 0; i < variance.length ; i++) {
            double sum =0;
            int k=0;
            for (int j = 0; j < patterns.length ; j++) {
                sum += Math.pow((patterns[j][i]-mean[i]),2);
                k+=1;

            }

            variance[i]= sum/k;
        }
        return variance;

    }


}
