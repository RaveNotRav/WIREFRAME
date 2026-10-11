// coordinates.js - Convert canvas clicks to natural image pixel coordinates

/**
 * Converts a click on a canvas (or image) into natural image coordinates.
 * Handles element scaling, object-fit letterboxing, and basic DPR (devicePixelRatio).
 * Note: EXIF orientation is assumed to be already applied by the browser's image decoder
 * (since image-orientation: from-image is standard). If the backend did not apply EXIF,
 * we would need inverse mapping. Here we assume the backend reads EXIF correctly.
 * 
 * @param {MouseEvent} event 
 * @param {HTMLImageElement|HTMLCanvasElement} element 
 * @param {number} naturalWidth 
 * @param {number} naturalHeight 
 * @param {string} objectFit - typically 'contain' or 'cover'
 * @returns { x, y } in natural pixel space
 */
export function getNaturalCoordinates(event, element, naturalWidth, naturalHeight, objectFit = 'contain') {
    const rect = element.getBoundingClientRect();
    
    // Client coordinates relative to the element
    const clientX = event.clientX - rect.left;
    const clientY = event.clientY - rect.top;

    if (objectFit === 'contain') {
        const scaleX = rect.width / naturalWidth;
        const scaleY = rect.height / naturalHeight;
        const scale = Math.min(scaleX, scaleY);

        const renderedWidth = naturalWidth * scale;
        const renderedHeight = naturalHeight * scale;

        const offsetX = (rect.width - renderedWidth) / 2;
        const offsetY = (rect.height - renderedHeight) / 2;

        const x = (clientX - offsetX) / scale;
        const y = (clientY - offsetY) / scale;

        // Clamp to natural bounds
        return {
            x: Math.max(0, Math.min(naturalWidth, x)),
            y: Math.max(0, Math.min(naturalHeight, y))
        };
    }
    
    // Default fallback (stretch / 100%)
    const scaleX = naturalWidth / rect.width;
    const scaleY = naturalHeight / rect.height;
    return {
        x: clientX * scaleX,
        y: clientY * scaleY
    };
}
