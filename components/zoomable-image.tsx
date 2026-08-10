"use client";

import { useEffect, useState } from "react";

interface ZoomableImageProps {
  src: string;
  alt: string;
  className?: string;
}

/**
 * Figure image that opens full size on click. Dense diagrams get scaled well
 * below their intrinsic width by the reading column, so the overlay shows the
 * image at natural size inside a scrollable container rather than fitting it
 * to the viewport — the point of zooming is to read the labels.
 */
export function ZoomableImage({ src, alt, className }: ZoomableImageProps) {
  const [isZoomed, setIsZoomed] = useState(false);

  useEffect(() => {
    if (!isZoomed) {
      return;
    }

    const handleKeyDown = (event: KeyboardEvent) => {
      if (event.key === "Escape") {
        setIsZoomed(false);
      }
    };

    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.addEventListener("keydown", handleKeyDown);

    return () => {
      document.body.style.overflow = previousOverflow;
      document.removeEventListener("keydown", handleKeyDown);
    };
  }, [isZoomed]);

  return (
    <>
      <button
        type="button"
        onClick={() => setIsZoomed(true)}
        aria-label={`View full size: ${alt}`}
        className="block w-full cursor-zoom-in"
      >
        {/* eslint-disable-next-line @next/next/no-img-element */}
        <img src={src} alt={alt} className={className} />
      </button>

      {isZoomed ? (
        <div
          role="dialog"
          aria-modal="true"
          aria-label={alt}
          onClick={() => setIsZoomed(false)}
          className="fixed inset-0 z-50 cursor-zoom-out overflow-auto bg-background/95 p-6 backdrop-blur-sm"
        >
          <div className="flex min-h-full min-w-full items-center justify-center">
            {/* eslint-disable-next-line @next/next/no-img-element */}
            <img src={src} alt={alt} className="max-w-none" />
          </div>
        </div>
      ) : null}
    </>
  );
}
