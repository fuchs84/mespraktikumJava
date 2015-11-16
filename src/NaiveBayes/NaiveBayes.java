package NaiveBayes;

import java.util.*;

/**
 * Created by Sebastian on 13.11.2015.
 */
public class NaiveBayes {
    public double[][] trainMatrix;
    public double[] trainlabel;
    public double[][][] featureprobability;

    public NaiveBayes(){
        trainMatrix = new double[0][0];
        trainlabel = new double[0];
    }

    public void addTrainData(double[][] pattern, double[] label){
        double [][] temppattern = trainMatrix;
        double [] templabel = trainlabel;
        trainMatrix = new double[trainMatrix.length+pattern.length][pattern[0].length];
        trainlabel = new double[trainlabel.length+label.length];
        for (int i = 0; i < temppattern.length; i++) {
            trainlabel[i] = templabel[i];
            for (int j = 0; j < trainMatrix[0].length; j++) {
                trainMatrix[i][j] = temppattern[i][j];
            }

        }

        for (int i = temppattern.length; i < temppattern.length+pattern.length; i++) {
            trainlabel[i] = label[i-temppattern.length];
            for (int j = 0; j < trainMatrix[0].length; j++) {
                trainMatrix[i][j] = pattern[i-temppattern.length][j];
            }

        }


    }
    public void train() {

        Integer[] numbers = new Integer[trainlabel.length];
        for (int i = 0; i < trainlabel.length; i++) {
            numbers[i] = (int) trainlabel[i];
        }
        Set<Integer> uniqKeys = new LinkedHashSet<Integer>(Arrays.asList(numbers ));

        System.out.println(uniqKeys.size());
        uniqKeys.toArray();
        Integer[] labelunique = uniqKeys.toArray(new Integer[uniqKeys.size()]);
        featureprobability = new double[uniqKeys.size()][trainMatrix[0].length+2][2];
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

            double[] mean = getmean(singlelabelpattern);
            double pmf = (double)k/trainlabel.length;
            double[] variance = getvariance(singlelabelpattern,pmf,mean);
            featureprobability[i][0][0] = labelunique[i];
            featureprobability[i][1][0] = pmf;
            for (int j = 2; j < featureprobability[0].length ; j++){

                featureprobability[i][j][0] = variance[j-2];
                featureprobability[i][j][1] = mean[j-2];
                System.out.println("label=  "+featureprobability[i][0][0]+"    variance:"+featureprobability[i][j][0]+"    mean:"+featureprobability[i][j][1]);

                }
            }

        }

    public double[] classifyalldata(double[][] testdata ){
        double[] predictedlabel = new double[testdata.length];
        for (int i = 0; i < predictedlabel.length; i++) {
            predictedlabel[i] = classifygaussian(testdata[i]);

        }
        return predictedlabel;
    }


    public double classifygaussian(double[] testdatapattern){
        double[] classprobability = new double[featureprobability.length];
        for (int i = 0; i <featureprobability.length ; i++) {
            double[] featureinclass = new double[testdatapattern.length];

            for (int j = 0; j <testdatapattern.length ; j++) {

                //gausswrskt
                featureinclass[j]=(1/Math.sqrt(2*Math.PI*Math.pow(featureprobability[i][j+2][0],2)))*
                                Math.pow(Math.E,
                                    -(Math.pow(testdatapattern[j]-featureprobability[i][j+2][1],2)/
                                    (2*Math.pow(featureprobability[i][j+2][0],2))));
            }
            double entireprob = featureprobability[i][1][0];
            for (int j = 0; j < testdatapattern.length; j++) {
                if (featureinclass[j]!=0.0) {
                    entireprob *= featureinclass[j];
                }
            }
            classprobability[i]=entireprob;
        }
        double extremum = 0;
        for (int t = 0; t < 1; t++) {
            for (int z = 0; z < classprobability.length; z++) {
                System.out.println("wrskt"+ classprobability[z]+"        ");

                for (int w = z; w < classprobability.length; w++) {
                    if (classprobability[z] > classprobability[w]) {
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

    public double[] getmean(double[][] pattern){
        double[] mean = new double[pattern[0].length];
        for (int i = 0; i < mean.length ; i++) {
            double sum =0;
            for (int j = 0; j < pattern.length ; j++) {
                sum += pattern[j][i];
            }
            mean[i]= sum/pattern.length;
        }
        return mean;
    }

    public double[] getvariance(double[][] pattern,double pmf,double[]mean){
        double[] variance = new double[pattern[0].length];
        for (int i = 0; i < variance.length ; i++) {
            double sum =0;
            int k=0;
            for (int j = 0; j < pattern.length ; j++) {
                sum += Math.pow((pattern[j][i]-mean[i]),2);
                k+=1;

            }

            variance[i]= sum/k;
        }
        return variance;

    }


}
