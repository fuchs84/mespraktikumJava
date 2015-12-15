package SelectData;

import java.io.BufferedReader;
import java.io.FileNotFoundException;
import java.io.FileReader;
import java.io.IOException;
import java.util.ArrayList;

/**
 * Created by MatthiasFuchs on 13.12.15.
 */
public class ReadData {

    public Data readCSVs(String patternPath, String labelPath) {
        double[][] pattern;
        double[][] label;
        ArrayList<String> patternData = new ArrayList<>();
        ArrayList<String> labelData = new ArrayList<>();
        int lineIndex = 0;
        try {
            BufferedReader brP = new BufferedReader(new FileReader(patternPath));
            BufferedReader brL = new BufferedReader(new FileReader(labelPath));


            String patternLine;
            String labelLine;

            while ((patternLine = brP.readLine()) != null && (labelLine = brL.readLine()) != null) {
                patternData.add(patternLine);
                labelData.add(labelLine);
                lineIndex++;
            }

        } catch (FileNotFoundException e) {
            e.printStackTrace();
        } catch (IOException e) {
            e.printStackTrace();
        }

        int offset = 0;
        int offsetElan = 4;
        String[] labelParts = labelData.get(0).split("\t");
        String[] patternParts = patternData.get(0).split("\t");
        pattern = new double[lineIndex - offset][patternParts.length-offsetElan];
        label = new double[lineIndex - offset][labelParts.length-offsetElan];

        for(int i = offset; i < lineIndex; i++) {
            patternParts = patternData.get(i).split("\t");
            labelParts = labelData.get(i).split("\t");
            if(pattern[0].length > label[0].length) {
                for(int j = 0; j < pattern[0].length; j++) {
                    pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetElan]);
                    if(j < label[0].length) {
                        label[i-offset][j] = Double.parseDouble(labelParts[j+offsetElan]);
                    }
                }
            } else {
                for(int j = 0; j < label[0].length; j++) {
                    label[i-offset][j] = Double.parseDouble(labelParts[j+offsetElan]);
                    if(j < pattern[0].length) {
                        pattern[i-offset][j] = Double.parseDouble(patternParts[j+offsetElan]);
                    }
                }
            }
        }
        return new Data(label, pattern);
    }
}