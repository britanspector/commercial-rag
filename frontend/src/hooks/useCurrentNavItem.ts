import { useMemo } from 'react'
import { useLocation } from 'react-router-dom'

import { getNavItemByPath, type NavItem } from '../navigation'

export function useCurrentNavItem(): NavItem | undefined {
  const { pathname } = useLocation()
  return useMemo(() => getNavItemByPath(pathname), [pathname])
}
