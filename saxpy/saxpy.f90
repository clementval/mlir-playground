
program main
  implicit none
  integer, parameter :: n = 1024
  real :: x(n), y(n), a = 10.0

  x(:) = 1.0
  y(:) = 2.0

  call saxpy(x,y,n,a)
  print*,y(:)

contains

  subroutine saxpy(x,y,n,a)
    implicit none
    integer :: n, i
    real :: a, x(n), y(n)
    !$acc data create(x,y)
    !$acc update device(x,y)
    !$acc parallel loop
    do i = 1, n
      y(i) = a*x(i)+y(i)
    end do
    !$acc update host(y)
    !$acc end data
  end subroutine saxpy

end program
