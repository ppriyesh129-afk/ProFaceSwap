package com.profaceswap

import android.os.Bundle
import android.widget.Button
import android.widget.ImageView
import android.widget.TextView
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var targetImage: ImageView
    private lateinit var statusText: TextView

    private val targetPicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                targetImage.setImageURI(uri)
                statusText.text = "Target image selected"
            }
        }

    private val sourcePicker =
        registerForActivityResult(ActivityResultContracts.GetContent()) { uri ->
            if (uri != null) {
                statusText.text = "Source face selected"
            }
        }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        targetImage = findViewById(R.id.targetImage)
        statusText = findViewById(R.id.statusText)

        val selectTargetButton =
            findViewById<Button>(R.id.selectTargetButton)

        val selectSourceButton =
            findViewById<Button>(R.id.selectSourceButton)

        val swapButton =
            findViewById<Button>(R.id.swapButton)

        selectTargetButton.setOnClickListener {
            targetPicker.launch("image/*")
        }

        selectSourceButton.setOnClickListener {
            sourcePicker.launch("image/*")
        }

        swapButton.setOnClickListener {
            statusText.text = "AI face-swap engine will be added next"
        }
    }
}
