import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function SearchPage() {
  const item = getNavItemByPath('/search')!
  return <NavPage item={item} />
}
