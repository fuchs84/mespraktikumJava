package SelectData;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;

/**
 * Created by MatthiasFuchs on 06.11.15.
 */
public class NWData {

    /**
     * Methode liest eine CSV-Datei ein
     * @param path Pfad der CSV-Datei
     * @return
     */
    public Data readCSV(String path) {
        double[][] pattern = null;
        double[] label = null;

        FileReader fr = null;
        try {
            fr = new FileReader(path);
        } catch (FileNotFoundException e) {
            e.printStackTrace();
        }

        String line;
        BufferedReader bf;

        try {
            bf = new BufferedReader(fr);
            int rows = 0;
            while (bf.readLine() != null) {
                rows++;
            }

            fr = new FileReader(path);
            bf = new BufferedReader(fr);
            line = bf.readLine();
            String[] temp = line.split("\t") ;
            int columns = temp.length;


            fr = new FileReader(path);
            bf = new BufferedReader(fr);

            pattern = new double[rows][columns-2];
            label = new double[rows];


            for(int i = 0; i <rows; i++) {
                line = bf.readLine();
                temp = line.split("\t");
                for(int j = 2; j < columns; j++) {
                    pattern[i][j-2] = Double.parseDouble(temp[j]);
                }
                label[i] = Double.parseDouble(temp[1]);
            }

        } catch (IOException e) {
            e.printStackTrace();
        }

        return new Data(label, pattern);
    }
}
