import { GroupType } from "./group";

/** determine which group should receive next image sample */
export function dispatchGroup(options: {
  current: Record<GroupType, number>;
  target: Record<GroupType, number>;
}): GroupType {
  const { current, target } = options;

  // Determine active groups
  const group_types: GroupType[] =
    target["val"] === 0 ? ["train", "test"] : ["train", "test", "val"];

  // Validate input
  for (const g of group_types) {
    if (target[g] <= 0) {
      throw new Error(`Invalid input: target['${g}'] = ${target[g]}; Expected a value larger than 0`);
    }
  }

  const total_target = group_types.reduce((sum, g) => sum + target[g], 0);

  let best_group: GroupType = group_types[0];
  let best_error = Infinity;

  const total_current = group_types.reduce((sum, g) => sum + current[g], 0);

  for (const group_type of group_types) {
    if (current[group_type]===0) {
      best_group = group_type
      break
    }

    const simulated = { ...current };
    simulated[group_type] += 1;
    const new_total = total_current + 1;

    let error = 0;
    for (const grp_type of group_types) {
      const actual_ratio = simulated[grp_type] / new_total;
      const target_ratio = target[grp_type] / total_target;
      const diff = actual_ratio - target_ratio;
      error += diff * diff;
    }

    if (error < best_error) {
      best_error = error;
      best_group = group_type;
    }
  }

  return best_group;
}

export function* genDispatchGroupSequence(options: {
  //   target: Record<GroupType, number | undefined>;
  target: Record<GroupType, number>;
  total: number;
}) {
  let { target, total } = options;
  let current = {
    train: 0,
    val: 0,
    test: 0,
  };
  for (let i = 0; i < total; i++) {
    let group_type = dispatchGroup({ current, target });
    current[group_type]++;
    yield group_type;
  }
}
