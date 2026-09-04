package com.profaceswap

import android.graphics.Bitmap
import java.nio.FloatBuffer

object ImageUtils {

    fun resize(bitmap: Bitmap, size: Int): Bitmap {
        return Bitmap.createScaledBitmap(bitmap, size, size, true)
    }

    fun bitmapToFloatBuffer(bitmap: Bitmap): FloatBuffer {
        val w = bitmap.width
        val h = bitmap.height
        val buffer = FloatBuffer.allocate(3 * w * h)

        val pixels = IntArray(w * h)
        bitmap.getPixels(pixels, 0, w, 0, 0, w, h)

        for (p in pixels) {
            buffer.put(((p shr 16) and 0xFF) / 255f)
            buffer.put(((p shr 8) and 0xFF) / 255f)
            buffer.put((p and 0xFF) / 255f)
        }

        buffer.rewind()
        return buffer
    }
}
