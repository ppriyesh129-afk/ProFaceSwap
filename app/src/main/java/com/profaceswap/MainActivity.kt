package com.profaceswap

import android.os.Bundle
import android.widget.TextView
import androidx.appcompat.app.AppCompatActivity

class MainActivity : AppCompatActivity() {

    private lateinit var statusText: TextView

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        setContentView(R.layout.activity_main)

        statusText = findViewById(R.id.statusText)

        statusText.text = "Inspecting AI models..."

        Thread {

            try {

                val engine =
                    FaceSwapEngine(this@MainActivity)

                val basicModelsLoaded =
                    engine.loadModels()

                if (!basicModelsLoaded) {

                    runOnUiThread {
                        statusText.text =
                            "Basic model inspection failed"
                    }

                    return@Thread
                }

                val hyperSwapLoaded =
                    engine.loadHyperSwap()

                runOnUiThread {

                    statusText.text =
                        if (hyperSwapLoaded) {
                            "AI model inspection complete"
                        } else {
                            "HyperSwap inspection failed"
                        }
                }

                engine.close()

            } catch (e: Throwable) {

                runOnUiThread {

                    statusText.text =
                        "Inspection error: ${e.javaClass.simpleName}"
                }
            }

        }.start()
    }
}
