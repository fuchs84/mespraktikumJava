package SelectData;
import java.util.ArrayList;

/**
 * Created by Sebastian on 12.11.2015.
 */
public class Crossvalidation {

    public static ArrayList<double[][]> crossvalidate(double [][] inputpattern, double [] inputlabel, int splits){
        ArrayList<double[][]> patternlist = new ArrayList();
        ArrayList<double[]> labelist = new ArrayList();

        int lengthsplit = (inputlabel.length/splits);
        System.out.println(lengthsplit);
        System.out.println(inputpattern.length);
        for (int i = 0; i < splits; i++) {
            double pattern[][]= new double[lengthsplit][inputpattern[0].length];
            double label[]= new double[lengthsplit];
            int k = i*lengthsplit;
            for (int j = 0; j <lengthsplit ; j++) {
                for (int l = 0; l <inputpattern[0].length ; l++) {
                   pattern[j][l] = inputpattern[k+j][l];
                }
                label[j]=inputlabel[k+j];
            }
            patternlist.add(pattern);
            labelist.add(label);
        }
        return patternlist;
    }

    public static ArrayList<double[]> crossvalidatelabel(double [][] inputpattern, double [] inputlabel, int splits){
        ArrayList<double[][]> patternlist = new ArrayList();
        ArrayList<double[]> labelist = new ArrayList();

        int lengthsplit = inputlabel.length/splits;
        for (int i = 0; i < splits; i++) {
            double pattern[][]= new double[lengthsplit][inputpattern[0].length];
            double label[]= new double[lengthsplit];
            int k = i*lengthsplit;
            for (int j = 0; j <lengthsplit ; j++) {
                for (int l = 0; l <inputpattern[0].length ; l++) {
                    pattern[j][l] = inputpattern[k+j][l];
                }
                label[j]=inputlabel[k+j];

            }
            patternlist.add(pattern);
            labelist.add(label);
        }
        return labelist;
    }
}





