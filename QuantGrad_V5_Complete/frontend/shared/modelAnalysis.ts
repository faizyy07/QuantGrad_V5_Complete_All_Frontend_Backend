export type ModelDetailKind = "metric" | "score" | "field";

export type ModelDetail = {
  key: string;
  label: string;
  value: string;
  numericValue?: number;
  kind: ModelDetailKind;
};

export function modelFieldLabel(key: string) {
  return key
    .replace(/([a-z0-9])([A-Z])/g, "$1 $2")
    .replace(/[._-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim()
    .replace(/\b\w/g, character => character.toUpperCase());
}
