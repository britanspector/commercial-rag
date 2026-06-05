import type { NavItem } from '../navigation'
import { PageShell } from './PageShell'

interface NavPageProps {
  item: NavItem
}

export function NavPage({ item }: NavPageProps) {
  return (
    <PageShell
      title={item.label}
      description={item.description}
      apiEndpoints={item.apiEndpoints}
      plannedFeatures={item.plannedFeatures}
      comingSoon={item.comingSoon}
    />
  )
}
