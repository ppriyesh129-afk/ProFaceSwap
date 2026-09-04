package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer

class BlazeFaceDetector(private val session: OrtSession) {

    private val env = OrtEnvironment.getEnvironment()

    fun detect(bitmap: Bitmap): OrtSession.Result {
        val input = ImageUtils.resize(bitmap, 128)
        val buffer: FloatBuffer = ImageUtils.bitmapToFloatBuffer(input)

        val tensor = OnnxTensor.createTensor(
            env,
            buffer,
            longArrayOf(1, 3, 128, 128)
        )

        return session.run(
            mapOf(session.inputNames.first() to tensor)
        )
    }
}
