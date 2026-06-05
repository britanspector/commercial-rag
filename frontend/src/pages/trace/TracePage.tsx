import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function TracePage() {
  const item = getNavItemByPath('/trace')!
  return <NavPage item={item} />
}
