package com.example.tflite;

import androidx.appcompat.app.AppCompatActivity;
import org.tensorflow.lite.Interpreter;

import android.content.res.AssetFileDescriptor;
import android.os.Bundle;
import android.view.View;
import android.widget.Button;
import android.widget.EditText;
import android.widget.TextView;

import java.io.FileInputStream;
import java.io.IOException;
import java.nio.MappedByteBuffer;
import java.nio.channels.FileChannel;

public class MainActivity extends AppCompatActivity {

    Button button;
    EditText editText;
    TextView textView;

    Interpreter tflite;
    @Override
    protected void onCreate(Bundle savedInstanceState) {
        super.onCreate(savedInstanceState);
        setContentView(R.layout.activity_main);

        button = findViewById(R.id.button);
        editText = findViewById(R.id.editText);
        textView = findViewById(R.id.textView);

        try{
                tflite = new Interpreter(loadModelFile());
            }catch (Exception ex){
                ex.printStackTrace();
        }

        button.setOnClickListener(new View.OnClickListener() {
            @Override
            public void onClick(View view) {
                float prediction = doInference(editText.getText().toString());
                textView.setText(Float.toString(prediction));
            }
        });

    }

    public  float doInference(String inputString){
        float[] inputVal = new float[1];
        inputVal[0] = Float.valueOf(inputString);

        float[][] outputval = new float[1][1];

        tflite.run(inputVal,outputval);

        float inferredValue = outputval[0][0];

        return  inferredValue;
    }
    private MappedByteBuffer loadModelFile() throws IOException{
        AssetFileDescriptor fileDescriptor = this.getAssets().openFd("linear.tflite");
        FileInputStream inputStream = new FileInputStream((fileDescriptor.getFileDescriptor()));
        FileChannel fileChannel = inputStream.getChannel();
        long startoffset = fileDescriptor.getStartOffset();
        long declaredLength = fileDescriptor.getDeclaredLength();
        return  fileChannel.map(FileChannel.MapMode.READ_ONLY,startoffset,declaredLength);
    }
}
