package com.profaceswap

import ai.onnxruntime.OnnxTensor
import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtSession
import android.graphics.Bitmap
import java.nio.FloatBuffer

class FaceLandmarker(
    private val session: OrtSession
) {

    private val environment = OrtEnvironment.getEnvironment()

    companion object {
        private const val INPUT_SIZE = 256
        const val LANDMARK_COUNT = 478
    }

    fun detect(bitmap: Bitmap): Array<FloatArray> {

        val face = Bitmap.createScaledBitmap(
            bitmap,
            INPUT_SIZE,
            INPUT_SIZE,
            true
        )

        val pixels = IntArray(INPUT_SIZE * INPUT_SIZE)

        face.getPixels(
            pixels,
            0,
            INPUT_SIZE,
            0,
            0,
            INPUT_SIZE,
            INPUT_SIZE
        )

        val buffer =
            FloatBuffer.allocate(3 * INPUT_SIZE * INPUT_SIZE)

        // RGB, CHW, normalized to [0, 1]
        for (channel in 0..2) {
            for (pixel in pixels) {

                val value = when (channel) {
                    0 -> (pixel shr 16) and 0xFF
                    1 -> (pixel shr 8) and 0xFF
                    else -> pixel and 0xFF
                }

                buffer.put(value / 255.0f)
            }
        }

        buffer.rewind()

        val tensor = OnnxTensor.createTensor(
            environment,
            buffer,
            longArrayOf(
                1,
                3,
                INPUT_SIZE.toLong(),
                INPUT_SIZE.toLong()
            )
        )

        val result = session.run(
            mapOf(
                session.inputNames.first() to tensor
            )
        )

        val rawLandmarks = result[0].value

        val landmarks = when (rawLandmarks) {

            is Array<*> -> {

                val first = rawLandmarks[0]

                when (first) {

                    is FloatArray -> {
                        Array(LANDMARK_COUNT) { index ->
                            val offset = index * 3

                            floatArrayOf(
                                first[offset],
                                first[offset + 1],
                                first[offset + 2]
                            )
                        }
                    }

                    is Array<*> -> {
                        Array(LANDMARK_COUNT) { index ->
                            val point = first[index]

                            when (point) {
                                is FloatArray -> point
                                is DoubleArray ->
                                    FloatArray(point.size) {
                                        point[it].toFloat()
                                    }

                                else -> FloatArray(3)
                            }
                        }
                    }

                    else -> emptyArray()
                }
            }

            is FloatArray -> {
                Array(LANDMARK_COUNT) { index ->
                    val offset = index * 3

                    floatArrayOf(
                        rawLandmarks[offset],
                        rawLandmarks[offset + 1],
                        rawLandmarks[offset + 2]
                    )
                }
            }

            else -> emptyArray()
        }

        tensor.close()
        result.close()
        face.recycle()

        return landmarks
    }
}
