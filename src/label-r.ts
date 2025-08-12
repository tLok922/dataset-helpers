import { Keypoint } from "./co-r";
// Type predicates
function isKeypoint(value: unknown): value is Keypoint {
  const k = value as Keypoint;
  return (
    typeof k.x === "number" &&
    typeof k.y === "number" &&
    (k.visibility === "unannotated" ||
      k.visibility === "not_visible" ||
      k.visibility === "visible") &&
    isBetweenZeroAndOne(k.x) &&
    isBetweenZeroAndOne(k.y)
  );
}

// function isMultiLabelKeypoint(value: unknown): value is Keypoint {
//   const k = value as Keypoint;
//   return (
//     typeof k.x === "number" &&
//     typeof k.y === "number" &&
//     (k.visibility === 0 || k.visibility === 1 || k.visibility === 2)
//   );
// }

// export function isBoundingBox(value: unknown): value is BoundingBox {
//   const b = value as BoundingBox;
//   return (
//     typeof b.class_idx === "number" &&
//     isBetweenZeroAndOne(b.x) &&
//     isBetweenZeroAndOne(b.y) &&
//     isBetweenZeroAndOne(b.width) &&
//     isBetweenZeroAndOne(b.height)
//   );
// }

// export function isBoundingBoxWithKeypoints(
//   value: unknown
// ): value is BoundingBoxWithKeypoints {
//   const b = value as BoundingBoxWithKeypoints;
//   return (
//     isBoundingBox(b) &&
//     Array.isArray(b.keypoints) &&
//     b.keypoints.every(isKeypoint)
//   );
// }

// Validation utilities
function isBetweenZeroAndOne(value: number): boolean {
  return value >= 0 && value <= 1;
}

function validateCategoryId(options: {
  categoryId: number;
  nClass: number;
}): void {
  const { categoryId: class_idx, nClass: n_class } = options;
  if (class_idx < 0 || class_idx >= n_class) {
    throw new Error(
      `Invalid class index: receive ${class_idx} but expect a range of [0,${
        n_class - 1
      }]`
    );
  }
}

function validateBoundingBox(box: {
  x: number;
  y: number;
  width: number;
  height: number;
}): void {
  const { x, y, width, height } = box;
  if (!isBetweenZeroAndOne(x) || !isBetweenZeroAndOne(y)) {
    throw new Error(
      `Invalid bounding box coordinates: x=${x}, y=${y}. Expected range [0, 1].`
    );
  }

  if (!isBetweenZeroAndOne(width) || !isBetweenZeroAndOne(height)) {
    throw new Error(
      `Invalid bounding box size: width=${width}, height=${height}. Expected range [0, 1].`
    );
  }
}

type BoundingBoxLabelStringOptions = {
  categoryId: number;
  nClass: number;
  x: number;
  y: number;
  width: number;
  height: number;
};

type BoundingBoxWithKeypointsLabelStringOptions =
  BoundingBoxLabelStringOptions & {
    keypoints: Keypoint[];
    nKeypoints: number;
    hasVisibility: boolean;
  };

export function toYoloBoundingBoxLabelString(
  args: BoundingBoxLabelStringOptions
): string {
  const { categoryId, x, y, width, height, nClass } = args;
  validateCategoryId({ categoryId, nClass });
  validateBoundingBox(args);
  return `${categoryId} ${x} ${y} ${width} ${height}`;
}

export function toYoloBoundingBoxWithKeypointsLabelString(
  args: BoundingBoxWithKeypointsLabelStringOptions
): string {
  const { keypoints, nKeypoints, hasVisibility } = args;
  let label = toYoloBoundingBoxLabelString(args);

  if (nKeypoints !== keypoints.length) {
    throw new Error(
      `Number of keypoints (${keypoints.length}) does not match n_keypoints (${nKeypoints})`
    );
  }

  if (!keypoints.every(isKeypoint)) {
    throw new Error("Invalid keypoints data");
  }

  for (const keypoint of keypoints) {
    label += ` ${keypoint.x} ${keypoint.y}`;
    label += hasVisibility ? ` ${keypoint.visibility}` : ``;
  }
  return label;
}
