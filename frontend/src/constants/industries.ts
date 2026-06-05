export interface IndustryOption {
  value: string
  label: string
}

export const INDUSTRY_OPTIONS: IndustryOption[] = [
  { value: 'uploads', label: '默认（uploads）' },
  { value: 'semi-conductor', label: '半导体' },
  { value: 'power-electronics', label: '电力' },
  { value: 'e-commercial', label: '互联网电商' },
  { value: 'appliance', label: '白色家电' },
]

export function getIndustryLabel(value: string): string {
  return INDUSTRY_OPTIONS.find((item) => item.value === value)?.label ?? value
}
