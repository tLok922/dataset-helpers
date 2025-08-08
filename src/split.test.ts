import { dispatchGroup } from "./split";
import { GroupType } from "./group";
import { even } from "@beenotung/tslib";

function test() {
    const current: Record<GroupType, number> = {
    train: 13,
    test: 4,
    val: 2,
  };
  const target: Record<GroupType, number> = {
    train: 7,
    test: 2,
    val: 1,
  };
  const group_type = dispatchGroup({ current, target });
  console.log(group_type);
}

test();
