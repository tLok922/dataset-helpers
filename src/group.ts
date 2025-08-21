export type GroupType = "train" | "val" | "test";

export const group_types = ["train", "test", "val"] satisfies Array<GroupType>;

export const isGroupKey = (key: string): key is GroupType =>
  key === "train" || key === "test" || key === "val";
