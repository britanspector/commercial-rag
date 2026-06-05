import { NavPage } from '../../components/NavPage'
import { getNavItemByPath } from '../../navigation'

export function UploadPage() {
  const item = getNavItemByPath('/upload')!
  return <NavPage item={item} />
}
