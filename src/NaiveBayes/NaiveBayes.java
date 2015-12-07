package NaiveBayes;

import java.util.*;

/**
 * Created by Sebastian on 13.11.2015.
 */
public class NaiveBayes {
    /**
     * Trainmatrix: Die Matrix mit der die Daten trainiert werden
     * trainlabel: die zur Matrix zugehörigen Label
     * featureprobability: Matrix der Form
     * [i][0][0] Klassen der Label
     * [i][1][0] Klassenwahrscheinlichkeit p(Ci) Label
     * [i][2+*][0] Varianz der Features
     * [i][2+*][1] Mittelwert der Features
     *
     */
    public double[][] trainMatrix;
    public double[] trainlabel;
    public double[][][] featureprobability;

    /**
     * initialisieren der Trainingsmatrix und der Labelmatrix
     */
    public NaiveBayes(){
        trainMatrix = new double[0][0];
        trainlabel = new double[0];
    }

    /**
     * Berechnet die Varianzen und Means der gegebenen Features innerhalb der einzelnen Klassen
     */
    public void train(double[][] pattern, double[] label) {
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

        Integer[] numbers = new Integer[trainlabel.length];
        for (int i = 0; i < trainlabel.length; i++) {
            numbers[i] = (int) trainlabel[i];
        }
        Set<Integer> uniqKeys = new LinkedHashSet<Integer>(Arrays.asList(numbers ));

        //erstellt dein Array mit den auftretenden labeln
        Integer[] labelunique = uniqKeys.toArray(new Integer[uniqKeys.size()]);
        featureprobability = new double[uniqKeys.size()][trainMatrix[0].length+2][2];
        //splitte die datenmatrix und Label auf die einzelnen Label auf
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

            //Berechnet die Varianz und Mittelwert auf den Trainingsdaten und weißt sie der Feature Wahrscheinlichkeit zu
            double[] mean = getmean(singlelabelpattern);
            double pmf = (double)k/trainlabel.length;
            double[] variance = getvariance(singlelabelpattern,mean);
            featureprobability[i][0][0] = labelunique[i];
            featureprobability[i][1][0] = pmf;
            for (int j = 2; j < featureprobability[0].length ; j++){

                featureprobability[i][j][0] = variance[j-2];
                featureprobability[i][j][1] = mean[j-2];
                System.out.println("label=  "+featureprobability[i][0][0]+"    variance:"+featureprobability[i][j][0]+"    mean:"+featureprobability[i][j][1]);

                }
            }


        }

    /**
     * Klassifiziert eine gegebene Feature-Matrix
     * @param testdata Matrix von Testdaten
     * @return Array der vorhergesagten Label
     */
    public double[] classify(double[][] testdata ){
        double[] predictedlabel = new double[testdata.length];
        for (int i = 0; i < predictedlabel.length; i++) {
            predictedlabel[i] = classifygaussian(testdata[i]);

        }
        return predictedlabel;
    }

    /**
     * Klassifiziert einzelnen Feature Vektor
     * @param testdatapattern Feature Vektor einer zu klassifizierenden Matrix
     * @return vorhergesagtes Label
     */
    public double classifygaussian(double[] testdatapattern){
        double[] classprobability = new double[featureprobability.length];
        for (int i = 0; i <featureprobability.length ; i++) {
            double[] featureinclass = new double[testdatapattern.length];

            for (int j = 0; j <testdatapattern.length ; j++) {

                //Berechnen der Gauss Wahrscheinlichkeit

                double proba =((1/Math.sqrt(2*Math.PI*Math.pow(featureprobability[i][j+2][0],2)))*
                                Math.pow(Math.E,
                                    -(Math.pow(testdatapattern[j]-featureprobability[i][j+2][1],2)/
                                    (2*Math.pow(featureprobability[i][j+2][0],2)))));

                if (proba!=0.0&&proba>0) {
                    featureinclass[j] = Math.abs(Math.log10(proba));
                }
            }
            double entireprob = featureprobability[i][1][0];
            for (int j = 0; j < testdatapattern.length; j++) {
                if (featureinclass[j]!=0.0&&featureinclass[j]>0) {
                    entireprob *= featureinclass[j];
               }
            }
            classprobability[i]=entireprob;
            System.out.println(entireprob+" "+i+"   Klasse: "+featureprobability[i][0][0]+"   Klassenwahrscheinlichkeit:"+featureprobability[i][1][0]);
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
     * Berechnet den Mittelwert der einzelnen Features innerhalb eines Labels
     * @param pattern Trainingsmatrix die in addTraindata eingelesen wurde
     * @return Mean array der features
     */
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

    /**
     * Berechnet die Varianz der einzelnen Features innerhalb eines Labels
     * @param pattern Trainingsmatrix die in addTraindata eingelesen wurde
     * @param mean der Mittelwert array Features
     * @return Varianz array der features
     */
    public double[] getvariance(double[][] pattern,double[]mean){
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
